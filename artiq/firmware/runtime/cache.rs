use alloc::{vec, boxed::Box, string::String, collections::btree_map::BTreeMap};
use core::{mem::size_of};
use refcounting::{RefAwareArray, RefCounted};

struct Entry {
    /// Owns the variable-sized allocation containing a
    /// `RefCounted<RefAwareArray<i32>>` followed by its trailing elements.
    ///
    /// `RefAwareArray` declares its trailing data as `[T; 0]`, so storing the
    /// struct directly would not reserve space for the runtime-sized elements.
    /// NAC3 aligns this array layout to 8 bytes on both 32-bit and 64-bit targets,
    /// so `u64` storage is used to preserve that alignment.
    data: Box<[u64]>,
    len: u32,
    borrowed: bool
}

impl Entry {
    fn array(&self) -> &RefCounted<RefAwareArray<i32>> {
        unsafe { &*(self.data.as_ptr() as *const RefCounted<RefAwareArray<i32>>) }
    }
}

impl core::fmt::Debug for Entry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let data = unsafe { self.array().inner.as_slice(self.len as usize) };
        f.debug_struct("Entry")
         .field("data", &data)
         .field("borrowed", &self.borrowed)
         .finish()
    }
}

// Copy the kernel-owned backing array into persistent cache-owned storage
fn copy_cache_data(source: &RefCounted<RefAwareArray<i32>>,
                   len: u32) -> Box<[u64]> {
    let size = size_of::<RefCounted<RefAwareArray<i32>>>() + len as usize * size_of::<i32>();
    let words = (size + size_of::<u64>() - 1) / size_of::<u64>();

    let mut storage = vec![0u64; words].into_boxed_slice();

    unsafe {
        let destination = &mut *(storage.as_mut_ptr() as *mut RefCounted<RefAwareArray<i32>>);
        destination.header.refcount = 0;
        destination.header.typeinfo_offset = 0;
        destination.inner.refcounted_elems = source.inner.refcounted_elems;
        destination
            .inner
            .as_mut_slice(len as usize)
            .copy_from_slice(source.inner.as_slice(len as usize));
    }
    storage
}

pub struct Cache {
    entries: BTreeMap<String, Entry>,
    empty: Box<RefCounted<RefAwareArray<i32>>>
}

impl core::fmt::Debug for Cache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cache")
         .field("entries", &self.entries)
         .finish()
    }
}

impl Cache {
    pub fn new() -> Cache {
        Cache { entries: BTreeMap::new(),
                empty: Box::new(RefCounted::empty_array()) }
    }

    pub fn get(&mut self, key: &str) -> (&RefCounted<RefAwareArray<i32>>, u32) {
        match self.entries.get_mut(key) {
            None => (
                &(*self.empty),
                0
            ),
            Some(entry) => {
                entry.borrowed = true;
                let len = entry.len;
                (entry.array(), len)
            }
        }
    }

    pub fn put(
        &mut self,
        key: &str,
        data: &RefCounted<RefAwareArray<i32>>,
        len: u32
    ) -> Result<(), ()> {
        match self.entries.get_mut(key) {
            None => (),
            Some(entry) => {
                if entry.borrowed { return Err(()) }
                entry.data = copy_cache_data(data, len);
                entry.len = len;
                return Ok(())
            }
        }

        self.entries.insert(String::from(key), Entry {
            data: copy_cache_data(data, len),
            len,
            borrowed: false
        });
        Ok(())
    }

    pub unsafe fn unborrow(&mut self) {
        for (_key, entry) in self.entries.iter_mut() {
            entry.borrowed = false;
        }
    }
}
