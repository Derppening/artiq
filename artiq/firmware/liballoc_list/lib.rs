#![no_std]

use core::{ptr, mem, fmt};
use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;

const MAGIC_FREE: usize = 0xDEADDEAD;
const MAGIC_BUSY: usize = 0xFEEDFEED;

// Headers are kept aligned to their own size and block sizes are multiples of
// it, so payloads are aligned to the header size without padding blocks.
#[derive(Debug)]
struct Header {
    magic: usize,
    size:  usize,
    // Next block in memory order, busy or free.
    next:  *mut Header,
    _pad:  usize
}

// Free list links, stored at the start of a free block's payload. Every block
// has a payload of at least one header, which is large enough to hold them.
struct Links {
    next:  *mut Header,
    prev:  *mut Header
}

pub struct ListAlloc {
    // All blocks, in memory order within each region.
    root:  *mut Header,
    // Free blocks only, most recently freed first.
    free:  Cell<*mut Header>
}

pub const EMPTY: ListAlloc = ListAlloc {
    root: 0 as *mut Header,
    free: Cell::new(0 as *mut Header)
};

unsafe fn links(header: *mut Header) -> *mut Links {
    header.offset(1) as *mut Links
}

/// Occupancy of a `ListAlloc`. Block headers are counted with the block they
/// precede, so `busy + idle` is the total size of all added regions, after
/// trimming them to header alignment.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    /// Bytes in busy blocks, headers included.
    pub busy: usize,
    /// Bytes in free blocks, headers included. Alignment padding is a free
    /// block and is counted here.
    pub idle: usize,
    /// Largest request a single allocation could satisfy right now, ignoring
    /// alignment padding. Runs of adjacent free blocks count as one, since
    /// `alloc` joins them, and the result is rounded down to a multiple of the
    /// header size, since `alloc` rounds requests up to one.
    pub largest_free: usize,
}

impl ListAlloc {
    pub unsafe fn stats(&self) -> Stats {
        let header_size = mem::size_of::<Header>();
        let mut stats = Stats { busy: 0, idle: 0, largest_free: 0 };
        // payload of the current run of free blocks, as one block after joining
        let mut run: Option<usize> = None;

        let mut curr = self.root;
        while !curr.is_null() {
            match (*curr).magic {
                MAGIC_BUSY => {
                    stats.busy += header_size + (*curr).size;
                    run = None;
                }
                MAGIC_FREE => {
                    stats.idle += header_size + (*curr).size;
                    let joined = match run {
                        Some(payload) => payload + header_size + (*curr).size,
                        None => (*curr).size,
                    };
                    stats.largest_free = stats.largest_free.max(joined - joined % header_size);
                    run = Some(joined);
                }
                _ => panic!("heap corruption detected at {:p}", curr)
            }
            curr = (*curr).next;
        }

        stats
    }

    pub unsafe fn add(&mut self, ptr: *mut u8, size: usize) {
        let header_size = mem::size_of::<Header>();
        let padding = (header_size - ptr as usize % header_size) % header_size;
        if size < padding + header_size * 2 { return }
        let size = (size - padding) / header_size * header_size;

        let curr = ptr.offset(padding as isize) as *mut Header;
        (*curr).magic = MAGIC_FREE;
        (*curr).size  = size - header_size;
        (*curr).next  = self.root;
        self.root = curr;
        self.push_free(curr);
    }

    pub unsafe fn add_range(&mut self, begin: *mut u8, end: *mut u8) {
        self.add(begin, end as usize - begin as usize)
    }

    unsafe fn push_free(&self, header: *mut Header) {
        let head = self.free.get();
        (*links(header)).next = head;
        (*links(header)).prev = ptr::null_mut();
        if !head.is_null() {
            (*links(head)).prev = header;
        }
        self.free.set(header);
    }

    unsafe fn insert_free_after(&self, after: *mut Header, header: *mut Header) {
        let next = (*links(after)).next;
        (*links(header)).next = next;
        (*links(header)).prev = after;
        if !next.is_null() {
            (*links(next)).prev = header;
        }
        (*links(after)).next = header;
    }

    unsafe fn unlink_free(&self, header: *mut Header) {
        let Links { next, prev } = ptr::read(links(header));
        if prev.is_null() {
            self.free.set(next);
        } else {
            (*links(prev)).next = next;
        }
        if !next.is_null() {
            (*links(next)).prev = prev;
        }
    }
}

unsafe impl GlobalAlloc for ListAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let header_size = mem::size_of::<Header>();
        let size;
        if layout.size() < header_size {
            // The payload must be able to hold the free list links once freed
            size = header_size
        } else if layout.size() % header_size != 0 {
            size = layout.size() + header_size - (layout.size() % header_size);
        } else {
            size = layout.size()
        }
        let align = layout.align();

        unsafe fn split(header: *mut Header, split_size: usize) -> *mut Header {
            let offset = mem::size_of::<Header>() + split_size;
            let next = (header as *mut u8).offset(offset as isize) as *mut Header;
            (*next).magic = MAGIC_FREE;
            (*next).size  = (*header).size - offset;
            (*next).next  = (*header).next;
            (*header).next  = next;
            (*header).size  = split_size;
            next
        }

        let mut curr = self.free.get();
        while !curr.is_null() {
            if (*curr).magic != MAGIC_FREE {
                panic!("heap corruption detected at {:p}", curr)
            }

            let mut next = (*curr).next;
            while !next.is_null() && (*next).magic == MAGIC_FREE {
                // Join
                self.unlink_free(next);
                (*next).magic = 0;
                (*curr).size += (*next).size + header_size;
                (*curr).next  = (*next).next;
                next = (*curr).next;
            }

            // Case 1: Memory can be allocated straight from the current chunk
            if (curr.offset(1) as usize) % align == 0 {
                // Check available space
                if (*curr).size >= size {
                    if (*curr).size > size + header_size * 2 {
                        let rest = split(curr, size);
                        self.insert_free_after(curr, rest);
                    }

                    self.unlink_free(curr);
                    (*curr).magic = MAGIC_BUSY;
                    return curr.offset(1) as *mut u8
                }
            }

            // Case 2: Padding is needed to satisfy the alignment
            else {
                let alloc_addr = curr.offset(2) as usize;
                let padding_size = align - (alloc_addr % align);

                if (*curr).size >= size + padding_size + header_size {
                    // Create a padding region, which stays in the free list
                    let block = split(curr, padding_size);

                    // Check if a padding is needed at the rear
                    if (*block).size > size + header_size * 2 {
                        let rest = split(block, size);
                        self.insert_free_after(curr, rest);
                    }

                    (*block).magic = MAGIC_BUSY;
                    return block.offset(1) as *mut u8
                }
            }

            curr = (*links(curr)).next;
        }

        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        let curr = (ptr as *mut Header).offset(-1);
        if (*curr).magic != MAGIC_BUSY {
            panic!("heap corruption detected at {:p}", curr)
        }
        (*curr).magic = MAGIC_FREE;
        self.push_free(curr);
    }
}

impl fmt::Display for ListAlloc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let mut total_busy = 0;
            let mut total_idle = 0;
            let mut total_meta = 0;

            let mut curr = self.root;
            while !curr.is_null() {
                total_meta += mem::size_of::<Header>();

                let desc = match (*curr).magic {
                    MAGIC_FREE => { total_idle += (*curr).size; "IDLE" },
                    MAGIC_BUSY => { total_busy += (*curr).size; "BUSY" },
                    _ => "!!!!"
                };

                write!(f, "{} {:p} + {:#x} + {:#x} -> {:p}\n",
                       desc, curr, mem::size_of::<Header>(), (*curr).size, (*curr).next)?;
                match (*curr).magic {
                    MAGIC_FREE | MAGIC_BUSY => (),
                    _ => break
                }

                curr = (*curr).next;
            }

            write!(f, " === busy: {:#x} idle: {:#x} meta: {:#x} total: {:#x}\n",
                   total_busy, total_idle, total_meta,
                   total_busy + total_idle + total_meta)
        }
    }
}
