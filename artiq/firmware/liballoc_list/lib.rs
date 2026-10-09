#![no_std]

use core::{ptr, mem, fmt};
use core::alloc::{GlobalAlloc, Layout};
#[cfg(feature = "peak")]
use core::cell::Cell;

const MAGIC_FREE: usize = 0xDEADDEAD;
const MAGIC_BUSY: usize = 0xFEEDFEED;

#[derive(Debug)]
struct Header {
    magic: usize,
    size:  usize,
    next:  *mut Header
}

pub struct ListAlloc {
    root:  *mut Header,
    /// Busy blocks right now.
    #[cfg(feature = "peak")]
    now:   Cell<Usage>,
    /// Most of each field of `now` since creation or `reset_peak`.
    #[cfg(feature = "peak")]
    peak:  Cell<Usage>
}

pub const EMPTY: ListAlloc = ListAlloc {
    root: 0 as *mut Header,
    #[cfg(feature = "peak")]
    now:  Cell::new(Usage { bytes: 0, blocks: 0 }),
    #[cfg(feature = "peak")]
    peak: Cell::new(Usage { bytes: 0, blocks: 0 })
};

/// Busy blocks of a `ListAlloc`, as tracked with the `peak` feature.
#[cfg(feature = "peak")]
#[derive(Debug, Clone, Copy)]
pub struct Usage {
    /// Bytes in busy blocks, headers included, as `Stats::busy` counts them.
    pub bytes: usize,
    /// Busy blocks, i.e. live allocations.
    pub blocks: usize,
}

/// Occupancy of a `ListAlloc`. Block headers are counted with the block they
/// precede, so `busy + idle` is the total size of all added regions.
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

    /// Most busy bytes and most busy blocks since the allocator was created or
    /// `reset_peak` was last called. The two maxima are tracked separately and
    /// need not have been reached at the same time.
    #[cfg(feature = "peak")]
    pub fn peak(&self) -> Usage {
        self.peak.get()
    }

    /// Restarts `peak` from the busy blocks now.
    #[cfg(feature = "peak")]
    pub fn reset_peak(&self) {
        self.peak.set(self.now.get())
    }

    /// Records that the block at `header` has become busy, or free if `busy`
    /// is false. Its size must be final.
    #[inline(always)]
    unsafe fn track(&self, header: *const Header, busy: bool) {
        #[cfg(feature = "peak")]
        {
            let bytes = mem::size_of::<Header>() + (*header).size;
            let mut now = self.now.get();
            if busy {
                now.bytes += bytes;
                now.blocks += 1;
                let mut peak = self.peak.get();
                peak.bytes = peak.bytes.max(now.bytes);
                peak.blocks = peak.blocks.max(now.blocks);
                self.peak.set(peak);
            } else {
                now.bytes -= bytes;
                now.blocks -= 1;
            }
            self.now.set(now);
        }
        #[cfg(not(feature = "peak"))]
        let _ = (header, busy);
    }

    pub unsafe fn add(&mut self, ptr: *mut u8, size: usize) {
        let header_size = mem::size_of::<Header>();
        if size < header_size * 2 { return }

        let curr = ptr as *mut Header;
        (*curr).magic = MAGIC_FREE;
        (*curr).size  = size - header_size;
        (*curr).next  = self.root;
        self.root = curr;
    }

    pub unsafe fn add_range(&mut self, begin: *mut u8, end: *mut u8) {
        self.add(begin, end as usize - begin as usize)
    }
}

unsafe impl GlobalAlloc for ListAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let header_size = mem::size_of::<Header>();
        let size;
        if layout.size() % header_size != 0 {
            size = layout.size() + header_size - (layout.size() % header_size);
        } else {
            size = layout.size()
        }
        let align = layout.align();

        let mut curr = self.root;
        while !curr.is_null() {
            match (*curr).magic {
                MAGIC_BUSY => (),
                MAGIC_FREE => {
                    let mut next = (*curr).next;
                    while !next.is_null() && (*next).magic == MAGIC_FREE {
                        // Join
                        (*next).magic = 0;
                        (*curr).size += (*next).size + header_size;
                        (*curr).next  = (*next).next;
                        next = (*curr).next;
                    }

                    unsafe fn split(header: *mut Header, split_size: usize) {
                        let offset = mem::size_of::<Header>() + split_size;
                        let next = (header as *mut u8).offset(offset as isize) as *mut Header;
                        (*next).magic = MAGIC_FREE;
                        (*next).size  = (*header).size - offset;
                        (*next).next  = (*header).next;
                        (*header).next  = next;
                        (*header).size  = split_size;
                    }

                    // Case 1: Memory can be allocated straight from the current chunk
                    if (curr.offset(1) as usize) % align == 0 {
                        // Check available space
                        if (*curr).size > size + header_size * 2 {
                            split(curr, size);
                        }

                        if (*curr).size >= size {
                            self.track(curr, true);
                            (*curr).magic = MAGIC_BUSY;
                            return curr.offset(1) as *mut u8
                        }
                    }

                    // Case 2: Padding is needed to satisfy the alignment
                    else {
                        let alloc_addr = curr.offset(2) as usize;
                        let padding_size = align - (alloc_addr % align);

                        if (*curr).size >= size + padding_size + header_size {
                            // Create a padding region
                            split(curr, padding_size);

                            curr = (*curr).next;

                            // Check if a padding is needed at the rear
                            if (*curr).size > size + header_size * 2 {
                                split(curr, size);
                            }

                            self.track(curr, true);
                            (*curr).magic = MAGIC_BUSY;
                            return curr.offset(1) as *mut u8
                        }
                    }
                },
                _ => panic!("heap corruption detected at {:p}", curr)
            }

            curr = (*curr).next;
        }

        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        let curr = (ptr as *mut Header).offset(-1);
        if (*curr).magic != MAGIC_BUSY {
            panic!("heap corruption detected at {:p}", curr)
        }
        self.track(curr, false);
        (*curr).magic = MAGIC_FREE;
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
