use core::slice;
use mem::size_of;

/// The object header prefixed to all composite objects, i.e. structs and tuples.
#[repr(C)]
pub struct ObjectHeader {
    pub refcount: u32,
    pub typeinfo_offset: i32,
}

/// A wrapper struct for objects with an object header.
#[repr(C)]
pub struct RefCounted<T> {
    pub header: ObjectHeader,
    pub inner: T,
}

/// An array storing possibly-reference-counted elements.
///
/// Note that `refcounted_elems` may not match the actual number of elements in `data` if `T` is not
/// a reference-counted type. For instance, `T = i32` would have `refcounted_elems = 0`.
#[repr(C)]
pub struct RefAwareArray<T> {
    pub refcounted_elems: u32,
    _padding: [u8; 8usize.saturating_sub(size_of::<usize>())],
    pub data: [T; 0],
}

/// A Python list as represented in NAC3.
#[repr(C)]
pub struct RawList<T> {
    pub items: *mut RefCounted<RefAwareArray<T>>,
    pub len: u32,
}

pub type List<T> = RefCounted<RawList<T>>;

impl<T> List<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts((*self.inner.items).inner.data.as_ptr(), self.inner.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut((*self.inner.items).inner.data.as_mut_ptr(), self.inner.len as usize) }
    }
}

