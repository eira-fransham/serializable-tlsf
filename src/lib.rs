use positioned_io::{Cursor, ReadAt, WriteAt};
use serde::{Deserialize, Serialize};
use std::{alloc, io, mem, ops};

type SecondLevelBitmap = u8;
type Pointer = u32;
type Size = u32;
type FirstLevelBitmap = Size;

const fn log2(val: usize) -> usize {
    num_bits::<usize>() - val.leading_zeros() as usize
}

const fn num_bits<T>() -> usize {
    mem::size_of::<T>() * 8
}

const MAX_HEADER_SIZE: usize = mem::size_of::<BlockHeader>() + mem::size_of::<FreeBlockHeader>();
/// This is the header size, aligned to the next power of 2.
const MINIMUM_BLOCK_SIZE: usize = match MAX_HEADER_SIZE.count_ones() {
    0 | 1 => MAX_HEADER_SIZE,
    _ => 1 << (log2(MAX_HEADER_SIZE) + 1),
};

/// Number of logarithmically-distributed first-level buckets.
///
/// We restrict block size to be at minimum the size of the free block header, and so this calculates
/// the number of level two structures we need. We need one for every possible value of log2(size),
/// so that's one for every bit of the Size type, not including log2 for values less than the minimum
/// block size.
const NUM_LOGARITHMIC_BUCKETS: usize =
    num_bits::<FirstLevelBitmap>() - log2(MINIMUM_BLOCK_SIZE) - 1;

/// Number of linearly-distributed second-level buckets.
const NUM_LINEAR_BUCKETS: usize = num_bits::<SecondLevelBitmap>();

/// The base allocator trait, intended to be implemented for a buffer. This allows the user of the
/// library to have complete control over the buffer itself, so long as it uses the buffer correctly
/// and does not overwrite memory used by the allocator.
///
/// It is the responsibility of the user to store the length of a block and to supply the correct
/// allocated size (returned by `allocate`, not the requested size) when reallocating and freeing.
///
/// This mirrors std::alloc::Allocator, but is intended to be used in a mutable, single-threaded way
/// and to return abstract indices instead of concrete pointers, and so this trait has no unsafe
/// functions. Additionally, grow/shrink have been coalesced since in a safe setting there is no
/// benefit to splitting the methods. Zeroed allocation/reallocation is not implemented.
///
/// Implementors of this trait may not cause unsafety no matter the input, but they may leave their
/// metadata in an arbitrarily incorrect state when given bad input.
pub trait Allocator {
    type Error;

    fn init(&mut self, size: Pointer) -> Result<(), Self::Error>;
    fn allocate(
        &mut self,
        layout: alloc::Layout,
    ) -> Result<Option<ops::Range<Pointer>>, Self::Error>;
    fn realloc(
        &mut self,
        ptr: Pointer,
        old_layout: alloc::Layout,
        new_layout: alloc::Layout,
    ) -> Result<Option<ops::Range<Pointer>>, Self::Error>;
    fn free(&mut self, ptr: Pointer, layout: ops::Range<Pointer>) -> Result<(), Self::Error>;
}

fn first_level_index(size: Size) -> usize {
    (size.leading_zeros() as usize).min(NUM_LINEAR_BUCKETS - 1)
}

fn second_level_index(size: Size) -> usize {
    ((size << size.leading_zeros() as usize) >> log2(NUM_LINEAR_BUCKETS)) as usize
}

fn mutate<R, T, F>(mut readwrite: R, offset: u64, func: F) -> io::Result<()>
where
    R: ReadAt + WriteAt,
    T: Serialize + for<'a> Deserialize<'a>,
    F: FnOnce(&mut R, &mut T) -> io::Result<()>,
{
    let mut value: T = bincode::deserialize_from(Cursor::new_pos(&mut readwrite, offset))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    func(&mut readwrite, &mut value)?;

    bincode::serialize_into(Cursor::new_pos(&mut readwrite, offset), &value)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

impl<T> Allocator for T
where
    T: ReadAt + WriteAt,
{
    type Error = io::Error;

    fn init(&mut self, size: Size) -> Result<(), Self::Error> {
        let mut header = AllocatorHeader::new(size);
        let central_header_size = bincode::serialized_size(&header)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let block_header = BlockHeader {
            prev_physical: PointerMeta::new(
                0,
                Meta {
                    free_block: true,
                    last_physical_block: true,
                },
            ),
        };
        let block_header_size = bincode::serialized_size(&block_header)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        bincode::serialize_into(
            Cursor::new_pos(&mut *self, central_header_size),
            &block_header,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        bincode::serialize_into(
            Cursor::new_pos(&mut *self, central_header_size + block_header_size),
            &FreeBlockHeader::new(size - central_header_size as u32),
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let fli = first_level_index(size);
        let sli = second_level_index(size);

        header.level1[fli].buckets[sli] = central_header_size as u32;
        header.level1[fli].free_bitmap |= 1 << sli;
        header.free_bitmap |= 1 << fli;

        bincode::serialize_into(Cursor::new(&mut *self), &header)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(())
    }

    fn allocate(
        &mut self,
        layout: alloc::Layout,
    ) -> Result<Option<ops::Range<Pointer>>, Self::Error> {
        if layout.align() > MINIMUM_BLOCK_SIZE {
            // We should request a slightly larger block and then return a pointer aligned forwards.
            todo!();
        }

        mutate(&mut *self, 0, |this, header: &mut AllocatorHeader| {
            let bucket = {
                let bitmap_mask = FirstLevelBitmap::MAX
                    >> (num_bits::<FirstLevelBitmap>()
                        - first_level_index(layout.size() as Size)
                        - 1);

                let bitmap = header.free_bitmap & bitmap_mask;

                let bucket_index =
                    num_bits::<FirstLevelBitmap>() - bitmap.count_zeros() as usize - 1;

                &mut header.level1[bucket_index]
            };

            let head = {
                let bitmap_mask = SecondLevelBitmap::MAX
                    >> (num_bits::<SecondLevelBitmap>()
                        - first_level_index(layout.size() as Size)
                        - 1);

                let bitmap = bucket.free_bitmap & bitmap_mask;

                let head_index =
                    num_bits::<SecondLevelBitmap>() - bitmap.count_zeros() as usize - 1;

                mem::take(&mut bucket.buckets[head_index])
            };

            mutate(
                &mut *this,
                head as u64,
                |this, block_header: &mut BlockHeader| {
                    let header_size = bincode::serialized_size(&block_header)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                    let free_head: FreeBlockHeader = bincode::deserialize_from(Cursor::new_pos(
                        &mut *this,
                        head as u64 + header_size,
                    ))
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

                    // TODO: Splitting

                    if free_head.prev_free != 0 {
                        mutate(
                            &mut *this,
                            free_head.prev_free as u64 + header_size,
                            |_, free_last: &mut FreeBlockHeader| {
                                free_last.next_free = free_head.next_free;

                                Ok(())
                            },
                        )?;
                    }

                    if free_head.next_free != 0 {
                        mutate(
                            &mut *this,
                            free_head.next_free as u64 + header_size,
                            |_, free_last: &mut FreeBlockHeader| {
                                free_last.prev_free = free_head.prev_free;

                                Ok(())
                            },
                        )?;
                    }

                    Ok(())
                },
            )?;
            // TODO: Should be pretty easy to avoid reserialising the entire allocator header.

            Ok(())
        })?;

        todo!()
    }

    fn realloc(
        &mut self,
        ptr: Pointer,
        old_layout: alloc::Layout,
        new_layout: alloc::Layout,
    ) -> Result<Option<ops::Range<Pointer>>, Self::Error> {
        todo!()
    }

    fn free(&mut self, ptr: Pointer, layout: ops::Range<Pointer>) -> Result<(), Self::Error> {
        todo!()
    }
}

impl AllocatorHeader {
    fn new(size: Size) -> Self {
        Self {
            free_bitmap: todo!(),
            level1: todo!(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct AllocatorHeader {
    free_bitmap: FirstLevelBitmap,
    /// The first-level index is `size.leading_zeros().min(NUM_LOGARITHMIC_BUCKETS - 1)`, second-level index
    /// is `(size << size.leading_zeros()) >> log2(SLI)` where SLI is the number of linear buckets.
    ///
    /// First-level buckets are distributed large-to-small and second-level buckets are distributed
    /// small-to-large in order to reduce the work needed to calculate an index.
    ///
    /// This array contains pointers to the head of the doubly-linked list of free blocks,
    /// of the given bucket size. Buckets at the first level are based on floor(log2(size))
    ///
    /// This could be calculated using log2(total_allocatable_area) instead of num_bits::<FirstLevelBitmap>()
    /// but it would be at the cost of simplicity and would only gain a small amount of memory.
    level1: [Level2; NUM_LOGARITHMIC_BUCKETS],
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
struct Level2 {
    free_bitmap: SecondLevelBitmap,
    buckets: [Pointer; NUM_LINEAR_BUCKETS],
}

/// Combined 62-bit pointer with 2-bit flags.
///
/// 62 bits as we allocate aligned to MINIMUM_BLOCK_SIZE, so the lower log2(MINIMUM_BLOCK_SIZE) bits
/// of the pointer will always be 0. log2(MINIMUM_BLOCK_SIZE) should always be greater than or equal to 2.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointerMeta(Pointer);

impl PointerMeta {
    fn new(mut pointer: Pointer, meta: Meta) -> Self {
        debug_assert_eq!(pointer & (LAST_PHYSICAL_BIT | FREE_BIT), 0);

        if meta.last_physical_block {
            pointer |= LAST_PHYSICAL_BIT;
        }

        if meta.free_block {
            pointer |= FREE_BIT;
        }

        Self(pointer)
    }
}

/// Metadata about a block.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Meta {
    /// Whether this block is the final block in the allocatable area, and so we should not attempt to
    /// find the next block in order to merge.
    pub last_physical_block: bool,
    /// Whether this block is free.
    pub free_block: bool,
}

const LAST_PHYSICAL_BIT: Size = 0b10;
const FREE_BIT: Size = 0b01;

impl PointerMeta {
    /// The block's size. This includes the size of the header itself.
    pub fn pointer(&self) -> Pointer {
        self.0 & !(LAST_PHYSICAL_BIT | FREE_BIT)
    }

    /// Metadata about the block - see [Meta].
    pub fn meta(&self) -> Meta {
        Meta {
            last_physical_block: self.0 & LAST_PHYSICAL_BIT != 0,
            free_block: self.0 & FREE_BIT != 0,
        }
    }
}

/// Header for a block that the allocator has control over.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHeader {
    /// The previous block, as ordered by physical address. This is used to efficiently coalesce blocks.
    pub prev_physical: PointerMeta,
}

/// Additional information, only for free blocks. Directly after `BlockHeader` in the block.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreeBlockHeader {
    pub size: Size,
    /// The previous block in the free list of this size class.
    pub prev_free: Pointer,
    /// The next block in the free list of this size class.
    pub next_free: Pointer,
}

impl FreeBlockHeader {
    fn new(size: Size) -> Self {
        Self {
            size,
            prev_free: 0,
            next_free: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
