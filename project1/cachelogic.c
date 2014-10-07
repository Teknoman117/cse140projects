#include "tips.h"

/* The following two functions are defined in util.c */

/* finds the highest 1 bit, and returns its position, else 0xFFFFFFFF */
unsigned int uint_log2(word w); 

/* return random int from 0..x-1 */
int randomint( int x );

/*
  This function allows the lfu information to be displayed

    assoc_index - the cache unit that contains the block to be modified
    block_index - the index of the block to be modified

  returns a string representation of the lfu information
 */
char* lfu_to_string(int assoc_index, int block_index)
{
  /* Buffer to print lfu information -- increase size as needed. */
  static char buffer[9];
  sprintf(buffer, "%u", cache[assoc_index].block[block_index].accessCount);

  return buffer;
}

/*
  This function allows the lru information to be displayed

    assoc_index - the cache unit that contains the block to be modified
    block_index - the index of the block to be modified

  returns a string representation of the lru information
 */
char* lru_to_string(int assoc_index, int block_index)
{
  /* Buffer to print lru information -- increase size as needed. */
  static char buffer[9];
  sprintf(buffer, "%u", cache[assoc_index].block[block_index].lru.value);

  return buffer;
}

/*
  This function initializes the lfu information

    assoc_index - the cache unit that contains the block to be modified
    block_number - the index of the block to be modified

*/
void init_lfu(int assoc_index, int block_index)
{
  cache[assoc_index].block[block_index].accessCount = 0;
}

/*
  This function initializes the lru information

    assoc_index - the cache unit that contains the block to be modified
    block_number - the index of the block to be modified

*/
void init_lru(int assoc_index, int block_index)
{
  cache[assoc_index].block[block_index].lru.value = 0;
}

/*
  Return the block to replace
*/

unsigned int blockToReplace(unsigned int set)
{
    // If we failed to retrieve from cache, attempt to load it
    // Attempt to find an invalid spot
    for(unsigned int i = 0; i < assoc; i++)
    {
        if(cache[set].block[i].valid == INVALID)
        {
            return i;
        }
    }

    // If there were no invalid blocks, implement our replacement policy
    // random replacement
    if(policy == RANDOM)
    {
        return randomint(assoc);
    }

    // least frequently used
    else if (policy == LFU)
    {
        unsigned int lowest = 0xFFFFFFFF;
        unsigned int block = 0;
        for(int i = 0; i < assoc; i++)
        {
            if(cache[set].block[i].accessCount < lowest)
            {
                block = i;
                lowest = cache[set].block[i].accessCount;
            }
        }
        return block;
    }

    // least recently used
    else
    {
        unsigned int greatest = 0;
        unsigned int block = 0;
        for(int i = 0; i < assoc; i++)
        {
            if(cache[set].block[i].lru.value > greatest)
            {
                block = i;
                greatest = cache[set].block[i].lru.value;
            }
        }
        return block;
    }

    return 0xFFFFFFFF;
}

/* Access the DRAM */

void accessDRAMWrapper(address addr, byte *data, unsigned int bytes, WriteEnable flag)
{
    for(int j = 0; j < bytes; j += 4)
    {
        accessDRAM(addr + j, data + j, WORD_SIZE, flag);
    }
}

/*
  This is the primary function you are filling out,
  You are free to add helper functions if you need them

  @param addr 32-bit byte address
  @param data a pointer to a SINGLE word (32-bits of data)
  @param we   if we == READ, then data used to return
              information back to CPU

              if we == WRITE, then data used to
              update Cache/DRAM
*/
void accessMemory(address addr, word* data, WriteEnable we)
{
    /* handle the case of no cache at all - leave this in */
    if(assoc == 0)
    {
        accessDRAM(addr, (byte*)data, WORD_SIZE, we);
        return;
    }

    // Compute the bit sizes of different fields
    unsigned int oBits = uint_log2(block_size);
    unsigned int sBits = uint_log2(set_count);
    unsigned int tBits = 32 - (oBits + sBits);

    // Compute bitwise masks for the fields
    unsigned int oMask = (1 << oBits) - 1;
    unsigned int sMask = (1 << sBits) - 1;
    unsigned int tMask = (1 << tBits) - 1;

    // Get the values of the fields
    unsigned int offset = addr & oMask;
    unsigned int set = (addr >> oBits) & sMask;
    unsigned int tag = (addr >> (oBits + sBits)) & tMask;

    // Assoc block to access
    unsigned int block = 0;

    // Attempt to find the row in cache
    for(unsigned int i = 0; i < assoc; i++)
    {
        if(cache[set].block[i].tag == tag)
        {
            block = i;
            if(cache[set].block[i].valid == INVALID)
            {
                // compulsory miss
                goto load;
            }

            // cache hit
            goto service;
        }
    }

    // cache miss
    block = blockToReplace(set);

load:
    // If the block we are about to replace is dirty, replace it in memory
    if(cache[set].block[block].dirty == DIRTY && cache[set].block[block].valid == VALID)
    {
        // Compute the address
        address a = cache[set].block[block].tag << (oBits + sBits);
        a |= (set << oBits);
        accessDRAMWrapper(a, (byte *) cache[set].block[block].data, block_size, WRITE);
    }

    // Perform load from dram
    accessDRAMWrapper(addr & ~oMask, (byte *) cache[set].block[block].data, block_size, READ);
    cache[set].block[block].valid = VALID;
    cache[set].block[block].dirty = VIRGIN;
    cache[set].block[block].tag = tag;
    cache[set].block[block].accessCount = 0;
    cache[set].block[block].lru.value = 0;

service:

    // Modify usage information
    cache[set].block[block].accessCount++;
    for(int i = 0; i < assoc; i++)
    {
        cache[set].block[i].lru.value++;
    }
    cache[set].block[block].lru.value = 0;

    // If we are reading from the memory
    if(we == READ)
    {
        // Service the read
        memcpy((void *) data, (void *) ((byte *) cache[set].block[block].data + offset), sizeof(word));
    }

    // Otherwise we are writing to memory
    else
    {
        // Service the write
        memcpy((void *) ((byte *) cache[set].block[block].data + offset), (void *) data, sizeof(word));

        // If the replacement policy is write through, service dram
        if(memory_sync_policy == WRITE_THROUGH)
        {
            // We need to set the offset to zero to load the whole row
            accessDRAMWrapper(addr & ~oMask, (byte *) cache[set].block[block].data, block_size, WRITE);
        }

        // Otherwise we have dirty memory
        else
        {
            cache[set].block[block].dirty = DIRTY;
        }
    }
}
