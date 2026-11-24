"""KernelWiki artifact PR-13340 (NVIDIA/TensorRT-LLM)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/blockKey.h"
#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheType.h"
#include "tensorrt_llm/batch_manager/llmRequest.h" // TODO forward declare
#include "tensorrt_llm/batch_manager/radixBlockTree.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvc = tensorrt_llm::executor::kv_cache;

namespace tensorrt_llm::batch_manager::eviction_policy
{
class BaseEvictionPolicy;
} // namespace tensorrt_llm::batch_manager::eviction_policy

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

static constexpr SizeType32 kPrimaryLevel = 0;

static constexpr SizeType32 kSecondaryLevel = 1;

// Extra block buffer allocated for SWA to be able to always keep "window size"
// tokens held in the blocks.
static constexpr SizeType32 kSWAExtraBlock = 1;

class KVCacheBlock;
class BlockManager;
class KVCacheManager;
class KVCacheTransferManager;

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using BlockPtr = std::shared_ptr<KVCacheBlock>;
using FreeBlocksQueue = std::list<BlockPtr>;
using UniqueToken = tensorrt_llm::runtime::UniqueToken;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using LoraTaskIdType = tensorrt_llm::runtime::LoraTaskIdType;
using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
using CacheSaltIDType = tensorrt_llm::runtime::CacheSaltIDType;
using MmKey = tensorrt_llm::executor::MmKey;
using WindowSizeType = SizeType32;

template <typename T>
using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

//! \brief Split vector into list of blocks of given size.
//! \param vec vector to split
//! \param usableSize part of the vector that is processed
//! \param elementsPerBlock desired size of blocks
//! \param allowPartial whether to append a block smaller than `elementsPerBlock` at the end
//! \return list of blocks
template <typename T>
std::list<std::vector<T>> chopVectorIntoBlocks(
    std::vector<T> const& vec, SizeType32 usableSize, SizeType32 elementsPerBlock, bool allowPartial)
{
    TLLM_CHECK_WITH_INFO(
        usableSize <= static_cast<SizeType32>(vec.size()), "usableSize=%d > %ld=vec.size()", usableSize, vec.size());
    std::list<std::vector<T>> blockedVectors;
    auto const vecEnd = vec.begin() + usableSize;
    for (auto begin = vec.begin(); begin < vecEnd; begin += elementsPerBlock)
    {
        auto blockSize = std::min(elementsPerBlock, static_cast<SizeType32>(std::distance(begin, vecEnd)));
        auto end = begin + blockSize;
        if (blockSize == elementsPerBlock || allowPartial)
        {
            blockedVectors.emplace_back(begin, end);
        }
    }
    return blockedVectors;
}

struct LinearAttentionMetadata
{
    enum LinearCacheType : WindowSizeType
    {
        kRecurrentStates = static_cast<WindowSizeType>(0x80000001),
    };

    std::vector<SizeType32> linearLayerIndices;
    WindowSizeType cacheType;
    SizeType32 allRecurrentStatesBytes; // Sum of all states like ssm_state and conv_state (1 layer)
    SizeType32 statesSnapshotInterval;  // Only used for kRecurrentStates
    bool saveLastSnapshot;              // Take additional snapshot of recurrent states at the end of the input sequence

    // Optional: explicit number of placeholder blocks for this kRecurrentStates manager.
    // If set, overrides the automatic computation (fullAttention.primaryBlocks - this.primaryBlocks).
    std::optional<SizeType32> numPlaceholderBlocks;

    [[nodiscard]] bool shouldAllocateRecurrentStates(
        SizeType32 currentBlockEndTokenIdx, SizeType32 promptLen, SizeType32 tokensPerBlock) const
    {
        // Allocate the last full block for maximum reuse opportunity.
        if (saveLastSnapshot && (currentBlockEndTokenIdx / tokensPerBlock == promptLen / tokensPerBlock))
        {
            TLLM_LOG_DEBUG("Allocating recurrent states for block %d, reason: saveLastSnapshot",
                (currentBlockEndTokenIdx / tokensPerBlock - 1));
            return true;
        }

        // Allocate the block that contains the end of the current sequence to save the final state.
        if (currentBlockEndTokenIdx >= promptLen && currentBlockEndTokenIdx < promptLen + tokensPerBlock)
        {
            TLLM_LOG_DEBUG("Allocating recurrent states for block %d, reason: end of sequence",
                (currentBlockEndTokenIdx / tokensPerBlock - 1));
            return true;
        }

        // We have checked statesSnapshotInterval is multiple of mTokensPerBlock during WindowBlockManager
        // initialization.
        if ((statesSnapshotInterval > 0) && (currentBlockEndTokenIdx % statesSnapshotInterval == 0))
        {
            TLLM_LOG_DEBUG("Allocating recurrent states for block %d, reason: statesSnapshotInterval",
                (currentBlockEndTokenIdx / tokensPerBlock - 1));
            return true;
        }
        return false;
    }

    [[nodiscard]] bool hasRecurrentStatesCache() const
    {
        return hasRecurrentStatesCache(cacheType);
    }

    static constexpr bool hasRecurrentStatesCache(WindowSizeType encodedWindowSize)
    {
        return (static_cast<uint32_t>(encodedWindowSize) & static_cast<uint32_t>(LinearCacheType::kRecurrentStates))
            == static_cast<uint32_t>(LinearCacheType::kRecurrentStates);
    }

    static constexpr bool hasLinearCache(WindowSizeType encodedWindowSize)
    {
        return hasRecurrentStatesCache(encodedWindowSize);
    }

    [[nodiscard]] SizeType32 calcMaxMemoryBlocks(
        WindowSizeType encodedWindowSize, SizeType32 tokensPerBlock, size_t memoryBudget, SizeType32 numLayers) const
    {
        if (hasRecurrentStatesCache(encodedWindowSize))
        {
            TLLM_CHECK_WITH_INFO(
                encodedWindowSize == kRecurrentStates, "each pool must only serve one type of linear cache");
            TLLM_CHECK_WITH_INFO(statesSnapshotInterval % tokensPerBlock == 0,
                "statesSnapshotInterval must be multiple of tokensPerBlock");
            // take a snapshot every `blockAlignment` blocks.
            auto perBlockBytes = allRecurrentStatesBytes * numLayers;
            auto numDynamicBlocks = (memoryBudget / perBlockBytes);
            return static_cast<SizeType32>(numDynamicBlocks);
        }
        TLLM_THROW("Unknown linear cache type");
    }
};

struct WindowSizeMetadata
{
    SizeType32 allottedPrimaryBlocks;   // Number of primary blocks allotted to the windowSize
    SizeType32 allottedSecondaryBlocks; // Number of secondary blocks allotted to the windowSize
    SizeType32 absolutePoolsOffset;     // cumulative number of pools up to manager
    SizeType32 numPools;                // number of managed pools
    SizeType32 maxTokenNum;             // Maximum token length per sequence (TODO: account for streamLLM)
    SizeType32 maxBlocksPerSeq;         // Maximum number of blocks per sequence
    SizeType32 maxNumBlocks;            // Number of primary+secondary blocks allotted to the windowSize
    SizeType32 windowSize;
    bool isSWA;

    std::string toString()
    {
        return tensorrt_llm::common::fmtstr(
            "WindowSizeMetadata{ .allottedPrimaryBlocks=%d, .allottedSecondaryBlocks=%d, .absolutePoolsOffset=%d, "
            ".numPools=%d, .maxTokenNum=%d, .maxBlocksPerSeq=%d, .maxNumBlocks=%d, "
            ".windowSize=%d, .isSWA=%d }",
            allottedPrimaryBlocks, allottedSecondaryBlocks, absolutePoolsOffset, numPools, maxTokenNum, maxBlocksPerSeq,
            maxNumBlocks, windowSize, isSWA);
    }
};

using NextBlockMap = std::unordered_map<BlockKey, BlockPtr, BlockKeyHasher>;

struct KvCacheStats
{
    // Number of maximum available blocks in the primary memory pool. This is determined and set by available primary
    // memory. See calculateMaxNumBlocks for details.
    SizeType32 maxNumBlocks;
    // Number of free blocks in the primary memory pool.
    SizeType32 freeNumBlocks;
    // Number of used blocks in the primary memory pool. usedNumBlocks = maxNumBlocks - freeNumBlocks.
    SizeType32 usedNumBlocks;
    SizeType32 toksPerBlock;
    // Total number of blocks allocated by all requests.
    SizeType32 allocTotalBlocks;
    // Number of new blocks that were allocated.
    SizeType32 allocNewBlocks;
    // Number of blocks that were matched and reused.
    SizeType32 reusedBlocks;
    // Number of blocks that were not matched and not reused.
    SizeType32 missedBlocks;
    // Measuring the KV Cache reuse rate. cacheHitRate = reusedBlocks / (reusedBlocks + missedBlocks).
    float cacheHitRate;
    // Number of free blocks for every configured attention-window size.
    std::map<SizeType32, SizeType32> numFreeBlocksPerWindowSize;
    // GPU bytes allocated for KV-cache
    std::size_t allocatedBytes{};
};

/// @brief Per-iteration KV cache statistics. All delta counters represent changes since the last call to
/// getIterationStats(). Gauges are instantaneous snapshots.
struct KvCacheIterationStats
{
    // --- Instantaneous gauges ---
    // Primary (GPU) pool
    SizeType32 primaryMaxNumBlocks{0};
    SizeType32 primaryFreeNumBlocks{0};
    SizeType32 primaryUsedNumBlocks{0};
    // Secondary (host) pool
    SizeType32 secondaryMaxNumBlocks{0};
    SizeType32 secondaryFreeNumBlocks{0};
    SizeType32 secondaryUsedNumBlocks{0};

    // --- Per-iteration deltas (reset on each read) ---
    // Context phase: block allocation and reuse
    SizeType32 iterAllocTotalBlocks{0};
    SizeType32 iterAllocNewBlocks{0};
    SizeType32 iterReusedBlocks{0};        // = iterFullReusedBlocks + iterPartialReusedBlocks
    SizeType32 iterFullReusedBlocks{0};    // blocks fully matched in radix tree
    SizeType32 iterPartialReusedBlocks{0}; // blocks partially matched in radix tree
    SizeType32 iterMissedBlocks{0};
    float iterCacheHitRate{0.0f};
    // Generation phase: block allocation
    SizeType32 iterGenAllocBlocks{0};

    // Transfer traffic deltas — host ↔ GPU
    SizeType32 iterOnboardBlocks{0};
    std::size_t iterOnboardBytes{0};
    SizeType32 iterOffloadBlocks{0};
    std::size_t iterOffloadBytes{0};
    // Intra-device (GPU → GPU) block copies (e.g. partial reuse when source block has refs)
    SizeType32 iterIntraDeviceCopyBlocks{0};
    std::size_t iterIntraDeviceCopyBytes{0};
};

// Basic building block of a paged KV cache - a single
// cache block. This class just holds metadata, no pointers
// since it is reused across all layers.
class KVCacheBlock : public std::enable_shared_from_this<KVCacheBlock>
{
public:
    using IdType = std::int32_t;

    static constexpr IdType kCachedBlocksRootId = -1;

    //! Sentinel block ID used by SWA on-demand placeholder blocks (no-arg createPlaceholder()).
    //! Chosen near the minimum int32 so it remains safely negatable if accidentally normalized,
    //! while any accidental mAllBlocksById[id] lookup still produces an obvious out-of-range
    //! failure rather than silently aliasing a real block ID.
    //! Linear-attention placeholders continue to use negative per-slot IDs via the 2-arg
    //! createPlaceholder(IdType, SizeType32) overload; this sentinel is specific to the SWA
    //! path where placeholders are swapped in-place for evicted OOW blocks without an ID.
    static constexpr IdType kPlaceholderBlockId = std::numeric_limits<IdType>::min() + 1;

    explicit KVCacheBlock(IdType blockId, kernels::KVCacheIndex blockIdx, SizeType32 windowSize = -1);

    void startScheduling();

    [[nodiscard]] IdType getBlockId() const;

    [[nodiscard]] NextBlockMap getNextBlocks() const;

    //! \brief Wire this block into the shared lookup tree at the given node and window size.
    //! \details If the block is already attached to a different node, the old attachment is
    //! cleared first (its value slot is erased and cascade pruning fires upward). Then the
    //! block is stored as the value for \p windowSize in \p node.
    //! \param node    The lookup-tree node to attach to.
    //! \param windowSize Value key identifying this block's slot within the node.
    void attachToLookupNode(radix_block_tree::LookupNodePtr node, int windowSize);

    //! \brief Detach this block from the lookup tree.
    //! \details Clears the block's value slot in its current node and resets mLookupNode /
    //! mWindowSize to their null states. The Node cascade-prune logic then removes empty
    //! ancestor nodes automatically (bottom-up, stopping at the first non-empty ancestor).
    void detachFromLookupNode();

    //! \brief Initialize a dummy root block's lookup-node link.
    //! \details Stores this block as the value for \p windowSize in \p rootNode so that
    //! direct children can retrieve the root block via getPrevBlock(). Must be called once
    //! after constructing the mCachedBlocksRoot block.
    //! \param rootNode  Root node of the per-manager UnifiedBlockTree.
    //! \param windowSize Window size associated with this WindowBlockManager.
    void setAsRoot(radix_block_tree::LookupNodePtr rootNode, int windowSize);

    [[nodiscard]] kernels::KVCacheIndex::UnderlyingType getMemoryPoolBlockIndex() const;

    [[nodiscard]] bool isPrimary() const;

    void swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock);

    void incRefCount();

    void decRefCount();

    void decSchedulingRefCount();

    [[nodiscard]] bool hasRefs() const;

    [[nodiscard]] bool hasSchedulingRefs() const;

    void setBlockKey(BlockKey const& blockKey, bool isFull);

    BlockKey getBlockKey();

    [[nodiscard]] VecUniqueTokens const& getUniqueTokens() const;

    //! \brief Return the lookup-tree node this block is attached to, or nullptr if not cached.
    [[nodiscard]] radix_block_tree::LookupNodePtr getLookupNode() const
    {
        return mLookupNode;
    }

    //! \brief Return the parent block in the lookup tree.
    //! \details Navigates via mLookupNode->getParentNode()->getValue(mWindowSize).
    //! Returns nullptr when:
    //!   - The block is not attached to the tree (mLookupNode == nullptr), or
    //!   - This block IS the root (mLookupNode->getParentNode() returns nullptr).
    //! For direct children of the root, returns the root block (mCachedBlocksRoot)
    //! NOTE: return type is by value (not const&) because the result is computed on the fly.
    [[nodiscard]] BlockPtr getPrevBlock() const;

    BlockPtr const& getPrevBlockInSeq() const;

    void setPrevBlockInSeq(BlockPtr prevBlock);

    void addNextBlock(BlockKey const& blockKey, BlockPtr block);

    void removeNextBlock(BlockKey const& blockKey);

    //! \brief True if this block has no physical GPU memory.
    //! \details Placeholder blocks exist in the sequence's block list to preserve prefix-chain
    //! structure in the lookup tree without consuming pool memory. Used by Mamba / linear-
    //! attention layers: only snapshot-position blocks are real; intervening positions are
    //! placeholders. Placeholder blocks are excluded from the eviction pool.
    [[nodiscard]] bool isPlaceholder() const;

    //! \brief Create a placeholder KVCacheBlock with no GPU memory (linear-attention form).
    //! \details The placeholder holds a block ID for sequence bookkeeping but mIsPlaceholder
    //! is set so that getCacheBlockIndices returns a nil index and the eviction pool ignores it.
    //! Used by the linear-attention queue path (tryAllocatePlaceholderForLinearAttention).
    static BlockPtr createPlaceholder(IdType blockId, SizeType32 windowSize);

    //! \brief Create a placeholder KVCacheBlock with no GPU memory (SWA on-demand form).
    //! \details Used by WindowBlockManager::detachFrontBlock when an out-of-window block is
    //! swapped out of a sequence's allocated-blocks list. mIsPlaceholder is set and
    //! WindowBlockManager::setOffsets maps kPlaceholderBlockId to a nil index. The block
    //! ID is set to kPlaceholderBlockId so getBlockById can distinguish it from pooled
    //! linear-attention placeholders. No windowSize is needed because SWA placeholders are
    //! never attached to the lookup tree.
    static BlockPtr createPlaceholder();

    void detachDescendantsFromLookupTree();

    void freeBlockAndAllDescendants();

    //! \brief Find block matching blockKey. If allowPartial is true, the returned block may match only a prefix of
    //! blockKey.
    //! @return tuple of [partialMatch, numMatched, block], partialMatch is true if not all the tokens of the block were
    //! matched.
    [[nodiscard]] std::tuple<bool, SizeType32, BlockPtr> findMatchingBlock(
        BlockKey const& blockKey, bool enablePartialReuse, bool copyOnPartialReuse) const;

    //! \brief Free block from previous block if present.
    void freeLeafBlock();

    [[nodiscard]] bool isFull() const;

    [[nodiscard]] bool isShared() const;

    [[nodiscard]] bool isLeaf() const;

    void setPriority(executor::RetentionPriority priority);

    [[nodiscard]] executor::RetentionPriority getPriority() const;

    void setDurationMs(std::optional<std::chrono::milliseconds> durationMs);

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDurationMs() const;

    void setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime);

    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point::duration> getExpirationTime() const;

    void setHash(size_t hash);

    // set hash automatically from block key and previous block in sequence
    void setHash();

    size_t getHash() const;

    std::vector<MmKey> getExtraKeys() const;

private:
    // Linear ID of block independent of pool
    IdType mBlockId;

    // Index of block in memory pool backing this block
    // Choice of pool is encoded into the type
    kernels::KVCacheIndex mMemoryPoolBlockIndex;

    // Number of references to the block
    SizeType32 mRefCount;

    // Number of references to the block
    SizeType32 mSchedulingRefCount;

    // Key of this block in the lookup tree (the token prefix it represents)
    BlockKey mBlockKey;

    // Pointer to this block's node in the shared UnifiedBlockTree.
    // nullptr when the block is not cached for reuse.
    radix_block_tree::LookupNodePtr mLookupNode;

    // Window size slot this block occupies in mLookupNode->mValue.
    // std::numeric_limits<int>::max() when mLookupNode is nullptr (unattached sentinel;
    // valid sizes are >= 1 or kRecurrentStates (-1); the sentinel is intentionally illegal
    // so accidental use on an unattached block triggers observable failures).
    int mWindowSize;

    // True when this block has no physical GPU memory (Mamba placeholder).
    bool mIsPlaceholder;

    // Previous block in the physical allocation sequence, nullptr for first block.
    // Distinct from getPrevBlock() (which navigates the radix lookup tree)
    BlockPtr mPrevBlockInSeq;

    // Iterator pointing to this block in mFreeBlocks.
    std::optional<FreeBlocksQueue::iterator> mFreeBlockIterator;

    // Flag indicating if block is full
    bool mIsFull;

    // Priority of the block
    executor::RetentionPriority mPriority;
    // Duration that the block's priority level applies for
    std::optional<std::chrono::milliseconds> mDurationMs;
    // Expiration time of the block
    std::optional<std::chrono::steady_clock::time_point::duration> mExpirationTime;
    // Hash for the event manager
    size_t mHash;
};

class GenerationRequest
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    explicit GenerationRequest(LlmRequest::RequestIdType requestId, SizeType32 numTokens, SizeType32 beamWidth,
        std::map<SizeType32, WindowSizeMetadata> const& windowSizeToMetadata,
        executor::KvCacheRetentionConfig kvCacheRetentionConfig = executor::KvCacheRetentionConfig())
        : mRequestId(requestId)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mNumFrontBlocksRemoved(0)
        , mCurrentPrepopulatedPromptLen(std::numeric_limits<SizeType32>::max())
    {
        auto const numWindowSizes = windowSizeToMetadata.size();
        mCacheBlockIds.reserve(numWindowSizes);
        mCacheBlockIndices.reserve(numWindowSizes);
        for (auto const [windowSize, metadata] : windowSizeToMetadata)
        {
            mCacheBlockIds[windowSize] = std::vector<std::vector<KVCacheBlock::IdType>>(beamWidth);
            auto const numPools = metadata.numPools;
            auto const maxBlocks = metadata.maxBlocksPerSeq;
            mCacheBlockIndices[windowSize]
                = runtime::BufferManager::cpu(runtime::ITensor::makeShape({numPools, beamWidth, 2, maxBlocks}),
                    runtime::TRTDataType<tensorrt_llm::kernels::KVCacheIndex>::value);
            auto cacheBlockIdsRange
                = runtime::BufferRange<tensorrt_llm::kernels::KVCacheIndex>(*mCacheBlockIndices.at(windowSize));
            std::fill(cacheBlockIdsRange.begin(), cacheBlockIdsRange.end(),
                tensorrt_llm::kernels::KVCacheIndex{
                    std::numeric_limits<tensorrt_llm::kernels::KVCacheIndex::UnderlyingType>::max()});
        }
    }

    void addNewTokens(SizeType32 n)
    {
        mNumTokens += n;
    }

    void removeTokens(SizeType32 n)
    {
        TLLM_CHECK(n <= mNumTokens);
        TLLM_CHECK(mNumTokens - n >= 0);
        mNumTokens -= n;
    }

    [[nodiscard]] LlmRequest::RequestIdType getRequestId() const
    {
        return mRequestId;
    }

    [[nodiscard]] SizeType32 getNumTokens() const
    {
        return mNumTokens;
    }

    [[nodiscard]] SizeType32 getNumFrontBlocksRemoved() const
    {
        return mNumFrontBlocksRemoved;
    }

    [[nodiscard]] SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::vector<std::vector<SizeType32>> const& getCacheBlockIds(SizeType32 windowSize) const
    {
        return mCacheBlockIds.at(windowSize);
    }

    [[nodiscard]] runtime::ITensor& getCacheBlockIndices(SizeType32 windowSize)
    {
        return *(mCacheBlockIndices.at(windowSize));
    }

    [[nodiscard]] runtime::ITensor const& getCacheBlockIndices(SizeType32 windowSize) const
    {
        return *(mCacheBlockIndices.at(windowSize));
    }

    void addCacheBlock(SizeType32 windowSize, SizeType32 beamIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(windowSize).at(beamIdx).push_back(blockId);
    }

    void changeCacheBlock(
        SizeType32 windowSize, SizeType32 beamIdx, SizeType32 pagedBlockIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(windowSize).at(beamIdx).at(pagedBlockIdx) = blockId;
    }

    void clearCacheBlocks(SizeType32 windowSize)
    {
        for (auto& beamBlockIds : mCacheBlockIds.at(windowSize))
        {
            beamBlockIds.clear();
        }
        mNumFrontBlocksRemoved = 0;
    }

    void removeFrontBlock(SizeType32 windowSize)
    {
        ++mNumFrontBlocksRemoved;
    }

    void removeLastBlock(SizeType32 windowSize)
    {
        for (auto& beamBlockIds : mCacheBlockIds.at(windowSize))
        {
            beamBlockIds.pop_back();
        }
    }

    [[nodiscard]] executor::RetentionPriority getDecodeRetentionPriority() const
    {
        return mKvCacheRetentionConfig.getDecodeRetentionPriority();
    }

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDecodeDurationMs() const
    {
        return mKvCacheRetentionConfig.getDecodeDurationMs();
    }

    [[nodiscard]] executor::KvCacheTransferMode getTransferMode() const
    {
        return mKvCacheRetentionConfig.getTransferMode();
    }

    [[nodiscard]] std::string const& getDirectory() const
    {
        return mKvCacheRetentionConfig.getDirectory();
    }

    [[nodiscard]] SizeType32 getCurrentPrepopulatedPromptLen() const
    {
        return mCurrentPrepopulatedPromptLen;
    }

    void setCurrentPrepopulatedPromptLen(SizeType32 currentPrepopulatedPromptLen)
    {
        TLLM_CHECK_WITH_INFO(currentPrepopulatedPromptLen <= mCurrentPrepopulatedPromptLen,
            "currentPrepopulatedPromptLen must be updated non-increasingly due to the "
            "assumption that smaller window sizes have shorter or equal"
            " currentPrepopulatedPromptLen during multi-window batch allocation.");
        mCurrentPrepopulatedPromptLen = currentPrepopulatedPromptLen;
    }

private:
    // Request id of the sequence
    LlmRequest::RequestIdType mRequestId;
    // Current number of generated tokens
    SizeType32 mNumTokens;
    // Number of beams
    SizeType32 mBeamWidth;
    // List of block ids allocated per each window size, for each beam of the sequence
    std::unordered_map<SizeType32, std::vector<std::vector<KVCacheBlock::IdType>>> mCacheBlockIds;
    // Tensor of block indices allocated per each window size, for each beam of the sequence
    std::unordered_map<SizeType32, runtime::ITensor::SharedPtr> mCacheBlockIndices;
    // The retention priority to assign to decode blocks
    executor::KvCacheRetentionConfig mKvCacheRetentionConfig;
    // Number of front blocks removed from the sequence
    SizeType32 mNumFrontBlocksRemoved;
    // Set of used blocks by the sequence
    std::set<KVCacheBlock::IdType> mUsedBlocks;
    // Current prepopulated prompt length
    SizeType32 mCurrentPrepopulatedPromptLen;
};

// attach metadata to a pool pointer
class KVCacheBlockPool
{
public:
    SizeType32 numLayers;
    SizeType32 kvFactor;
    SizeType32 numKvHeads;
    SizeType32 sizePerHead;
    SizeType32 tokensPerBlock;
    SizeType32 blockSize;

    // Memory pools. Primary is fast memory, secondary is slower memory used for offloading.
    runtime::ITensor::SharedPtr primaryPtr;
    runtime::ITensor::SharedPtr secondaryPtr;

    // FP4 KV caches have extra pools that contain second level scales for dequantization.
    bool containsBlockScales;
    bool containsIndexerKCache;

    // When true, pool tensor is laid out as {numLayers, numBlocks, kvFactor, blockSize}
    // instead of the default {numBlocks, numLayers, kvFactor, blockSize}.
    // Used for recurrent state (linear attention) pools.
    bool layerFirstLayout;

    KVCacheBlockPool(SizeType32 numLayers, SizeType32 kvFactor, SizeType32 numKvHeads, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, runtime::ITensor::SharedPtr primaryPtr = nullptr,
        runtime::ITensor::SharedPtr secondaryPtr = nullptr, bool containsBlockScales = false,
        bool containsIndexerKCache = false)
        : numLayers(numLayers)
        , kvFactor(kvFactor)
        , numKvHeads(numKvHeads)
        , sizePerHead(sizePerHead)
        , tokensPerBlock(tokensPerBlock)
        , blockSize(numKvHeads * sizePerHead * tokensPerBlock)
        , primaryPtr(std::move(primaryPtr))
        , secondaryPtr(std::move(secondaryPtr))
        , containsBlockScales(containsBlockScales)
        , containsIndexerKCache(containsIndexerKCache)
        , layerFirstLayout(false)
    {
    }
};

/// \brief Result of a single radix tree walk that collects all prefix reuse information.
/// \details Replaces separate findNewContextBlock() + countReusableBlocks(true) +
///          countReusableBlocks(false) calls with a single walk, reducing 2-3 walks to 1.
struct PrefixReuseSummary
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    /// Number of prefix blocks already allocated (refCount > 0).
    /// Used by the block budget to avoid double-counting with the eviction free count.
    SizeType32 reusableBlocksAllocated{0};

    /// Total number of prefix blocks cached (allocated or free-cached).
    /// Used by the token budget (NoEvict) since all cached tokens avoid recompute.
    SizeType32 reusableBlocksAll{0};

    /// First block key NOT found in the radix tree. std::nullopt means either all full
    /// prefix blocks matched (full prefix hit) or the request has no full block key to
    /// probe yet; a concrete BlockKey identifies the first missing full block.
    /// Used by the capacity scheduler's skip-check logic to decide whether to defer a request.
    std::optional<BlockKey> firstNewBlock{std::nullopt};
};

// The WindowBlockManager manages the metadata of KVCacheBlocks.
// It manages multiple arrays of cache blocks called pools.
// Layers with the same number of kv heads are grouped under the same pool.
// Each pool has shape [max_blocks, num_layers, 2, num_kv_heads, tokens_pre_block, head_size], where num_layers refers
// to the number of layers with the same num_kv_heads that share that pool.
// The metadata of KVCacheBlocks is shared between layers, so each block spans all of the managed pool - an allocated
// block matches some chunk of memory in each pool. The shape of the chunk in every pool is [2, num_kv_heads,
// tokens_per_block, head_size]. The size per block and number of blocks are pre-determined and set in the constructor.
// WindowBlockManager maintains a list of free blocks at any time.
//
// FP4 KV caches allocate additional pools for block scale factors. These pools have the same
// shape as the regular KV pools, except that the the last dim is head_size / N where N is determined
// by the precise FP4 format being used (16 for NVFP4). There is one block scale pool per normal pool.
//
// BlockManager maintains a list of free blocks at any time.
// Alloc pops off the block at the front, and Free pushes it back to the vector.
// WindowBlockManager maintains a vector of lists of request ids to allocated blocks
// per sequence. This can be used to Free all blocks belonging to a sequence.
class WindowBlockManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;
    using BaseEvictionPolicy = tensorrt_llm::batch_manager::eviction_policy::BaseEvictionPolicy;
    using BlockMap = std::unordered_multimap<size_t, BlockPtr>;
    using BlockMapIterRange = std::pair<BlockMap::const_iterator, BlockMap::const_iterator>;

    explicit WindowBlockManager(nvinfer1::DataType dtype, SizeType32 windowSize,
        std::vector<SizeType32> const& managedLayers, std::vector<SizeType32> const& numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, bool isSWA, SizeType32 blocksInPrimaryPool,
        SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream,
        CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
        std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
        radix_block_tree::UnifiedBlockTree& lookupTree, std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent = nullptr,
        bool enableIndexerKCache = false, SizeType32 indexerKCacheQuantBlockSize = 128,
        SizeType32 indexerKCacheIndexHeadDim = 0, bool indexerKCacheUseFp4 = false,
        std::optional<LinearAttentionMetadata> linearAttentionMetadata = std::nullopt,
        SizeType32 numPlaceholderBlocks = 0);

    ~WindowBlockManager();

    [[nodiscard]] bool isEnableIndexerKCache() const
    {
        return mEnableIndexerKCache;
    }

    [[nodiscard]] SizeType32 getIndexerKCacheQuantBlockSize() const
    {
        return mIndexerKCacheQuantBlockSize;
    }

    [[nodiscard]] SizeType32 getIndexerKCacheIndexHeadDim() const
    {
        return mIndexerKCacheIndexHeadDim;
    }

    [[nodiscard]] bool getIndexerKCacheUseFp4() const
    {
        return mIndexerKCacheUseFp4;
    }

    void allocatePools(bool useUvm);

    void releasePools();

    void createIndexerKCachePools();

    void startScheduling();

    //! \brief Per-request block allocation statistics from batch addSequence.
    struct BatchSeqStats
    {
        SizeType32 prepopulatedLen{0};
        SizeType32 allocTotalDelta{0};
        SizeType32 allocNewDelta{0};
        SizeType32 reusedDelta{0};
        SizeType32 missedDelta{0};
    };

    //! \brief Result of Phase 1 (claim-only) of batch addSequence.
    //! \details Holds matched blocks and prepared data so Phase 2 can proceed without
    //!          re-traversing the radix tree.
    struct ClaimResult
    {
        struct ClaimedBlock
        {
            BlockPtr block{nullptr};
            SizeType32 numMatchedTokens{0}; //!< tokens matched in this block
            bool isPartialMatch{false};
            bool needsCopy{false};       //!< partial match on block with refs or non-leaf (needs getFreeBlock + copy)
            bool isPlaceholder{false};   //!< placeholder block (linear attention recurrent states)
            bool isTraversalOnly{false}; //!< SWA OOW anchor with a trie node but no cache block value
            bool shouldReleaseCopySource{false}; //!< last copier releases the claimed source after copy
        };

        std::vector<ClaimedBlock> claimedBlocks;
        SizeType32 totalMatchedTokens{0};
        SizeType32 latestMatchingNonPlaceholderBlockIdx{-1};
        SizeType32 numSharedContextBlocks{0};
        SizeType32 numContextBlocks{0};
        bool shareLastContextBlockAmongBeams{true};
        std::vector<BlockKey> blockKeys;
        std::vector<executor::RetentionPriorityAndDuration> perBlockRetentions;
        executor::KvCacheTransferMode mode{executor::KvCacheTransferMode::DRAM};
        std::string directory;
    };

    //! \brief Tracks which request currently "owns" a partially-matched leaf block across
    //!        the batch Phase 1 loop, so that at most one request reuses the block in-place
    //!        while all others copy.
    struct PartialClaimTracker
    {
        struct Entry
        {
            size_t requestIdx; //!< index of the request that currently owns the reuse
            size_t claimedIdx; //!< index into that request's claimedBlocks vector
            bool fullyMatched; //!< true once any request fully matches this block
        };

        //! Keyed by block ID.
        std::unordered_map<KVCacheBlock::IdType, Entry> map;
    };

    //! \brief Batch add sequences with two-phase claim-then-onboard under a single lock.
    //! \details Phase 1 claims all matching blocks across all requests (protecting from eviction).
    //!          Phase 2 onboards host blocks and allocates non-matching blocks.
    //!          The unified lookup tree's mutex is held for the entire operation.
    //! \param sequences  Per-request GenerationRequest references (parallel with other vectors).
    //! \param inputLengths  Per-request effective input length.
    //! \param numContextBlocksVec  Per-request number of context blocks.
    //! \param llmRequests  Per-request LlmRequest references.
    //! \return Per-request prepopulatedPromptLen.
    [[nodiscard]] std::vector<BatchSeqStats> addSequenceBatch(std::vector<GenerationRequest*> const& sequences,
        std::vector<SizeType32> const& inputLengths, std::vector<SizeType32> const& numContextBlocksVec,
        std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests, bool isEnableBlockReuse);

    //! \brief Allocate new block for each beam of the sequence.
    //! \details Might free cached blocks if no free blocks are available.
    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams);

    //! \brief According to request's current position, copy data from the last full block to the next block (ignoring
    //! the placeholder block). It should be called after every context chunk is processed.
    void copyLinearAttentionBlock(GenerationRequest& sequence, LlmRequest const& llmRequest);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx);

    [[nodiscard]] std::vector<KVCacheBlock::IdType> storeBlocksForReuse(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks = false);

    void storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest);

    //! \brief Pin blocks associated with a sequence to prevent eviction.
    void pinBlocks(GenerationRequest& sequence);

    //! \brief Release blocks of the sequence.
    //! \details When llmRequest is provided and reuse is enabled, blocks will be stored.
    std::optional<KVCacheBlock::IdType> releaseBlocks(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest);

    //! \brief Simulate freeing all blocks for that sequence to check impact on number of free blocks
    void schedulingReleaseBlocks(LlmRequest::RequestIdType requestId);

    //! \brief Update cache offsets for last block
    void updateLastCacheBlockOffsets(GenerationRequest& seq);

    //! \brief Release last block in the sequence
    void releaseLastBlock(GenerationRequest& sequence);

    //! \brief Detach front block from the sequence
    void detachFrontBlock(GenerationRequest& sequence);

    //! \brief Add/detach block(s) to/from the sequence if needed
    //! \details When we need a new block, we add it. For sliding window
    //! attention (SWA), when a block goes out-of-window (OOW), we detach it
    //! If this called in the first step of the generation phase, we may detach
    //! more than a single block since there may be more than one context block
    //! that goes OOW.
    void adjustBlocksIfNeeded(GenerationRequest& sequence);

    [[nodiscard]] SizeType32 getWindowSize() const noexcept
    {
        return mWindowSize;
    }

    [[nodiscard]] std::string const& getLogPrefix() const noexcept
    {
        return mLogPrefix;
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const;

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mAllocTotalBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mAllocNewBlocks;
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept
    {
        return mReusedBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocatedBlocks() const
    {
        return getMaxNumBlocks() - getNumFreeBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mMissedBlocks;
    }

    // Get num free blocks in the secondary (host) memory pool
    [[nodiscard]] SizeType32 getNumFreeSecondaryBlocks() const noexcept;

    //! \brief Get iteration stats (deltas since last call) for this window. Resets internal delta snapshots.
    [[nodiscard]] KvCacheIterationStats getAndResetIterationStats();

    [[nodiscard]] bool hasFreeBlocks(SizeType32 numRequired = 1) const
    {
        return getNumFreeBlocks() >= numRequired;
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired) const noexcept
    {
        return mSchedulingNumFreeBlocks >= numRequired;
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const noexcept
    {
        return static_cast<SizeType32>(mAllBlocksById.size());
    }

    [[nodiscard]] BlockPtr getBlockById(KVCacheBlock::IdType blockId) const;

    [[nodiscard]] SizeType32 getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    //! \brief Get size of one K/V cache block in one layer for the specified pool.
    //! @details Volume of [numKvHeads, tokensPerBlock, sizePerHead] in the specified pool.
    [[nodiscard]] SizeType32 getBlockSize(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).blockSize;
    }

    [[nodiscard]] SizeType32 getNumEltsPerContainer() const
    {
#ifdef ENABLE_FP4
        return mDataType == nvinfer1::DataType::kFP4 ? 2 : 1;
#else
        return 1;
#endif
    }

    [[nodiscard]] SizeType32 getNumPools(
        bool includeBlockScalePools = true, bool includeIndexerKCachePools = true) const noexcept
    {
        if (includeBlockScalePools && includeIndexerKCachePools)
        {
            return mPools.size();
        }
        SizeType32 count = 0;
        for (auto const& pool : mPools)
        {
            if (includeBlockScalePools && pool.containsBlockScales)
            {
                count++;
            }
            else if (includeIndexerKCachePools && pool.containsIndexerKCache)
            {
                count++;
            }
            if (!pool.containsBlockScales && !pool.containsIndexerKCache)
            {
                count++;
            }
        }
        return count;
    }

    [[nodiscard]] KVCacheBlockPool const& getPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx);
    }

    [[nodiscard]] bool containsBlockScales(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).containsBlockScales;
    }

    [[nodiscard]] SizeType32 getNumPrimaryBlocks() const
    {
        return mNumPrimaryBlocks;
    }

    [[nodiscard]] SizeType32 getNumSecondaryBlocks() const
    {
        return mNumSecondaryBlocks;
    }

    [[nodiscard]] SizeType32 getLayerPoolIdx(SizeType32 layerIdx) const
    {
        return mLayerToPoolIndex.at(layerIdx);
    }

    //! \brief Maps a global layer index to its layer index within its pool.
    //! \details If we only have one pool, then getPoolLayerIdx(i) == i. Otherwise,
    //! \details gives the layer index into the getLayerPoolIdx(i).
    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layerIdx) const
    {
        return mLayerToIndexWithinPool.at(layerIdx);
    }

    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId) const;

    //! \brief Bring offloaded block from secondary to primary memory.
    //! \details Does nothing if block is already in primary memory.
    void onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    //! \brief Bring block from primary to secondary memory.
    //! \details Does nothing if block is already in secondary memory.
    void offloadBlock(BlockPtr const& block, executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM,
        std::string const& directory = "");

    //! \brief Combined prefix reuse analysis — single radix tree walk.
    [[nodiscard]] PrefixReuseSummary analyzePrefixReuse(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    //! \brief Sync internal streams used by transfer manager with buffer manager stream
    void syncTransferManagerWithBufferManager();

    //! \brief Perform per-request bookkeeping
    void refreshBlocks();

    [[nodiscard]] static bool blockInRadixTree(BlockPtr const& block);

    //! \brief Store context blocks in the reuse trie for this window.
    //! \details Called after context phase for both SWA and non-SWA windows.
    //!          Must be called before any detachFrontBlock call so that OOW blocks
    //!          are already in the trie when they are replaced with placeholders.
    void storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest);

    //! \brief Store blocks in the reuse trie.
    //! \param blockKeys Key of each block.
    //! \param blocks Block pointers (beam 0 only). OOW slots contain placeholder blocks
    //!        (isPlaceholder()==true); storeBlocks advances past them via a trie lookup
    //!        when the anchor still exists, and continues past missing SWA anchors.
    //! \param pinBlocks If true, increment ref count for blocks while storing.
    //! \return Pair of (num blocks stored for reuse, vector of pinned block IDs).
    [[nodiscard]] std::pair<SizeType32, std::vector<KVCacheBlock::IdType>> storeBlocks(
        std::vector<BlockKey> blockKeys, std::vector<BlockPtr> const& blocks, bool pinBlocks = false);

    [[nodiscard]] bool verifyQueueIntegrity() const;

    //! \brief Return whether this window is SWA.
    [[nodiscard]] bool isSWA() const
    {
        return mIsSWA;
    }

    [[nodiscard]] bool isEnablePartialReuse() const
    {
        return mEnablePartialReuse;
    }

    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(BlockKey const& blockKey);

    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKeys(
        std::vector<BlockKey> const& blockKeys);

    //! \brief Unpin blocks by block ids directly
    void unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds);

    void resetReuseState()
    {
        std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());
        // The shared lookup tree is reset once by BlockManager::resetReuseState() before
        // this method is called.  Here we only need to re-create the per-window root block
        // and wire it into the (already fresh) shared tree.
        mCachedBlocksRoot = std::make_shared<KVCacheBlock>(KVCacheBlock::kCachedBlocksRootId,
            tensorrt_llm::kernels::KVCacheIndex{
                std::numeric_limits<tensorrt_llm::kernels::KVCacheIndex::UnderlyingType>::max()});
        mCachedBlocksRoot->setAsRoot(mLookupTree->getRoot(), mWindowSize);
    }

private:
    //! \brief Walk the reuse tree with precomputed per-block keys (no lock; callers must hold mLookupTree->getMutex()).
    [[nodiscard]] std::shared_ptr<KVCacheBlock> searchReuseTree(std::vector<BlockKey> const& blockKeys);

    struct ReuseMatch
    {
        BlockPtr block;
        SizeType32 numMatchedTokens{0};
        bool isPartialMatch{false};
        bool isTraversalOnly{false};
    };

    struct ReuseMatchResult
    {
        std::vector<ReuseMatch> matches;
        SizeType32 totalMatchedTokens{0};
        std::optional<BlockKey> firstNewBlock{std::nullopt};
    };

    //! \brief Find the reusable prefix, including SWA traversal-only anchors.
    //! \details Value-less exact nodes are only accepted for SWA when enough later tokens
    //!          are matched to place the missing anchor outside the attention window.
    [[nodiscard]] ReuseMatchResult findReusableBlockMatches(std::vector<BlockKey> const& blockKeys,
        bool enablePartialReuse, bool copyOnPartialReuse, SizeType32 maxMatchedTokens) const;

    bool tryAllocatePlaceholderForLinearAttention(GenerationRequest& sequence, bool shareAmongBeams);

    //! \brief Add single block to beam of sequence and mAllocatedBlocksPerSeq.
    void addBlockToBeam(BlockPtr const& block, GenerationRequest& sequence, SizeType32 beamIdx);

    //! \brief Add single block to all beams of sequence.
    void addBlockToAllBeams(BlockPtr const& block, GenerationRequest& sequence);

    //! \brief Phase 1: Walk radix tree and claim matching blocks.
    //! \details Caller must hold mLookupTree->getMutex().
    //!          Uses \p tracker to coordinate partial-match ownership across requests in
    //!          the same batch. \p claimResults is the full vector so that a previous
    //!          request's ClaimedBlock can be retroactively marked needsCopy.
    [[nodiscard]] ClaimResult claimMatchingBlocks(GenerationRequest& sequence, SizeType32 inputLength,
        SizeType32 numContextBlocks, LlmRequest& llmRequest, size_t requestIdx, PartialClaimTracker& tracker,
        std::vector<ClaimResult>& claimResults);

    //! \brief Build ClaimResult metadata without walking the radix tree.
    //! \details Used for non-reuse path where all blocks are freshly allocated.
    [[nodiscard]] ClaimResult buildClaimResultMetadata(
        GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest);

    //! \brief Phase 2: Onboard claimed host blocks and allocate non-matching blocks.
    //! \details Caller must hold mLookupTree->getMutex().
    [[nodiscard]] SizeType32 onboardAndAllocateBlocks(
        GenerationRequest& sequence, LlmRequest& llmRequest, ClaimResult& claimResult, bool isEnableBlockReuse);

    //! \brief Find block least likely to be reused, free it if necessary and return.
    //! \param sequence Sequence which the free block is allocated for
    //! \param wantPlaceholder If true, return a pre-allocated placeholder block instead of a normal block
    [[nodiscard]] BlockPtr getFreeBlock(GenerationRequest& sequence,
        executor::RetentionPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority,
        std::optional<std::chrono::milliseconds> durationMs = std::nullopt,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "",
        bool wantPlaceholder = false);

    //! \brief Calls KVCacheBlock::freeLeafBlock to remove block from search tree.
    void freeLeafBlock(BlockPtr const& block);

    //! \brief For FP4 quantization. Creates pool objects for FP4 block scalars.
    void createBlockScalePools(SizeType32 blockSize);

    //! \brief This WindowBlockManager is for holding SSM states for linear attention models.
    [[nodiscard]] bool isRecurrentState() const
    {
        return mLinearAttentionMetadata.has_value()
            && LinearAttentionMetadata::hasRecurrentStatesCache(mLinearAttentionMetadata->cacheType);
    }

private:
    nvinfer1::DataType mDataType;
    SizeType32 mWindowSize;

    // Number of blocks in pools
    SizeType32 mNumPrimaryBlocks;
    SizeType32 mNumSecondaryBlocks;

    // List of allocated blocks for each sequences
    std::unordered_map<LlmRequest::RequestIdType, std::vector<BlockPtr>> mAllocatedBlocksPerSeq;

    // Pool per unique numKvHeads in the model
    std::vector<KVCacheBlockPool> mPools;

    // Matching layers to their respective pools: {<layer #0>: <pool idx 2>, }, etc.
    std::unordered_map<SizeType32, SizeType32> mLayerToPoolIndex;
    // Matching layers to their index *within* their respective pools: {..., <layer 3>: <idx 2 within pool> }. See
    // getPoolLayerIdx
    std::unordered_map<SizeType32, SizeType32> mLayerToIndexWithinPool;

    // Buffer manager
    runtime::BufferManager mBufferManager;

    // Used to keep track of number of free blocks during scheduling
    SizeType32 mSchedulingNumFreeBlocks;
    // Number of tokens per one block
    SizeType32 mTokensPerBlock;
    // Whether this window is sliding window attention/full attention
    bool mIsSWA;
    // List of all blocks by idx
    std::vector<BlockPtr> mAllBlocksById;
    // Pre-allocated placeholder blocks for linear attention (recurrent state) managers.
    // Indexed by abs(blockId): mAllPlaceholderBlocksById[abs(blockId)] gives the block with that negative ID.
    // Indices 0 and 1 are unused (nullptr); valid blocks start at index 2 (blockId == -2).
    std::vector<BlockPtr> mAllPlaceholderBlocksById;
    // Pointer to the shared radix lookup tree owned by BlockManager.
    // All WindowBlockManager instances under the same BlockManager share one tree,
    // using window size as the value key so their nodes coexist in the same trie.
    radix_block_tree::UnifiedBlockTree* mLookupTree;
    // Dummy block acting as root for BlockToken searches
    BlockPtr mCachedBlocksRoot;
    // KV cache type (self or cross)
    CacheType mCacheType;
    // Eviction Policy
    std::shared_ptr<BaseEvictionPolicy> mEvictionPolicy;
    // Event manager
    std::shared_ptr<KVCacheEventManager> mEventManager;
    // Pointer to parent loopback agent
    std::shared_ptr<kvc::BaseLoopbackAgent> mLoopbackAgent;
    // Transfer manager
    std::shared_ptr<KVCacheTransferManager> mTransferManager;

    // Statistics for block allocations/reuse
    // Total number of blocks allocated by all requests (context phase)
    SizeType32 mAllocTotalBlocks;
    // Number of new blocks that were allocated (context phase)
    SizeType32 mAllocNewBlocks;
    // Number of blocks that were fully reused (context phase)
    SizeType32 mFullReusedBlocks;
    // Number of blocks that were partially reused (context phase)
    SizeType32 mPartialReusedBlocks;
    // Number of blocks that were reused (full + partial, context phase)
    SizeType32 mReusedBlocks;
    // Number of unique blocks that were reused
    SizeType32 mReusedUniqueBlocks;
    // Number of blocks that were not reused (context phase)
    SizeType32 mMissedBlocks;
    // Number of blocks allocated during generation phase
    SizeType32 mGenAllocBlocks;
    // Only be 1 or 2. If 2: general KV stored. If 1: K == V for any token, so only K is stored to optimize the
    // max_num_tokens(For DeepSeek). Controlled by mCacheType
    SizeType32 mKVFactor;
    std::set<KVCacheBlock::IdType> mReusedBlockIds;
    std::string const mLogPrefix;
    // Number of reused tokens
    double mReusedTokens;
    // Total number of input tokens
    double mTotalInputTokens;
    // Whether blocks that are partially matched should be reused.
    bool mEnablePartialReuse;
    // Whether partially matched blocks that are already in use should be copied and reused.
    bool mCopyOnPartialReuse;
    // The kv cache connector manager
    std::shared_ptr<kv_connector::KvCacheConnectorManager> mKvCacheConnectorManager;

    // Snapshot of cumulative counters at last iteration stats read (for delta computation)
    SizeType32 mPrevAllocTotalBlocks{0};
    SizeType32 mPrevAllocNewBlocks{0};
    SizeType32 mPrevReusedBlocks{0};
    SizeType32 mPrevFullReusedBlocks{0};
    SizeType32 mPrevPartialReusedBlocks{0};
    SizeType32 mPrevMissedBlocks{0};
    SizeType32 mPrevGenAllocBlocks{0};

    // Whether to enable indexer K cache
    bool mEnableIndexerKCache;
    // Quant block size for indexer K cache
    SizeType32 mIndexerKCacheQuantBlockSize;
    // Index head dim for indexer K cache
    SizeType32 mIndexerKCacheIndexHeadDim;
    // Whether the indexer K cache stores FP4-packed data (half the byte count
    // per token vs. FP8). Drives the createIndexerKCachePools() formula.
    bool mIndexerKCacheUseFp4{false};

    std::optional<LinearAttentionMetadata> mLinearAttentionMetadata;
};

class BlockManager
{
public:
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using BaseEvictionPolicy = tensorrt_llm::batch_manager::eviction_policy::BaseEvictionPolicy;

    explicit BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
        CudaStreamPtr stream, SizeType32 maxSequenceLength, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype, SizeType32 sinkBubbleLength,
        SizeType32 chunkSize, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnPartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr,
        std::optional<kvc::BaseAgentConfig> agentConfig = std::nullopt, bool enableIndexerKCache = false,
        SizeType32 indexerKCacheQuantBlockSize = 128, SizeType32 indexerKCacheIndexHeadDim = 0,
        bool indexerKCacheUseFp4 = false,
        std::optional<LinearAttentionMetadata> linearAttentionMetadata = std::nullopt);

    [[nodiscard]] bool isEnableIndexerKCache() const
    {
        return mIsEnableIndexerKCache;
    }

    [[nodiscard]] SizeType32 getIndexerKCacheQuantBlockSize() const
    {
        return mIndexerKCacheQuantBlockSize;
    }

    [[nodiscard]] SizeType32 getIndexerKCacheIndexHeadDim() const
    {
        return mIndexerKCacheIndexHeadDim;
    }

    [[nodiscard]] bool getIndexerKCacheUseFp4() const
    {
        return mIndexerKCacheUseFp4;
    }

    [[nodiscard]] bool isEnablePartialReuse() const
    {
        return mWindowBlockManagers.begin()->second.isEnablePartialReuse();
    }

    BlockManager(BlockManager const&) = delete;
    BlockManager& operator=(BlockManager const&) = delete;

    //! \brief Calculate the proportional share each window size receives of the total memory pool
    //! \details Example:       (uniqueWindowSizeToLayers={1024: [1], 4096: [0, 4, 5], 8192: [2, 3]})
    //!          Would Return:  {1024: 0.0345, 4096: 0.4138, 8192: 0.5517} [sums to 1.0].
    //!          See: TEST_F(KVCacheManagerTest, BlockManagerTestWindowSizeToShare).
    //! \return Map<windowSize, share> where share is a float between 0 and 1. Shares sum to 1.0.
    static std::map<SizeType32, float> calculateWindowSizeToShare(
        std::map<SizeType32, std::vector<SizeType32>> const& uniqueWindowSizeToLayers,
        std::map<SizeType32, SizeType32> const& cacheSizePerTokenPerWindowSize);

    void allocatePools(bool useUvm);

    //! \brief Batch add sequences forwarding to WindowBlockManager::addSequenceBatch.
    [[nodiscard]] std::vector<WindowBlockManager::BatchSeqStats> addSequenceBatch(
        std::vector<GenerationRequest*> const& sequences, std::vector<SizeType32> const& inputLengths,
        std::vector<SizeType32> const& numContextBlocksVec,
        std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests, SizeType32 windowSize,
        bool isEnableBlockReuse);

    void allocateBlock(GenerationRequest& sequence, SizeType32 windowSize);

    //! \brief According to request's current position, copy data from the last full block to the next block (ignoring
    //! the placeholder block). It should be called after every context chunk is processed.
    void copyLinearAttentionBlock(GenerationRequest& sequence, LlmRequest const& llmRequest);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx);

    std::optional<KVCacheBlock::IdType> releaseBlocks(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinBlocks = false);

    [[nodiscard]] std::vector<KVCacheBlock::IdType> storeBlocksForReuse(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinBlocks = false);

    void schedulingReleaseBlocks(LlmRequest::RequestIdType requestId);

    /// @brief Pin all blocks associated with a sequence across all window managers.
    /// @param sequence The generation request whose blocks should be pinned.
    void pinBlocks(GenerationRequest& sequence);

    void unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds);

    void releaseLastBlock(GenerationRequest& sequence, SizeType32 windowSize);

    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId, SizeType32 windowSize) const;

    //! \brief Combined prefix reuse analysis — single radix tree walk.
    //! \details WILL NOT WORK FOR VARIABLE WINDOW ATTENTION.
    [[nodiscard]] PrefixReuseSummary analyzePrefixReuse(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    //! \brief Bring block from primary to secondary memory for window size.
    //! \details Does nothing if block is already in primary memory.
    void onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock, SizeType32 windowSize,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    //! \brief Bring block from primary to secondary memory for window size.
    //! \details Does nothing if block is already in secondary memory.
    void offloadBlock(BlockPtr const& block, SizeType32 windowSize,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    [[nodiscard]] std::pair<SizeType32, std::vector<KVCacheBlock::IdType>> storeBlocks(std::vector<BlockKey> blockKeys,
        std::vector<BlockPtr> const& blocks, SizeType32 windowSize, bool pinBlocks = false)
    {
        return mWindowBlockManagers.at(windowSize).storeBlocks(std::move(blockKeys), blocks, pinBlocks);
    }

    [[nodiscard]] bool verifyQueueIntegrity(SizeType32 windowSize) const;

    void releasePools();

    void startScheduling();

    [[nodiscard]] std::map<SizeType32, SizeType32> getNumFreeBlocksPerWindowSize() const
    {
        std::map<SizeType32, SizeType32> numFreeBlocksPerWindowSize;
        for (auto const& [windowSize, manager] : mWindowBlockManagers)
        {
            numFreeBlocksPerWindowSize[windowSize] = manager.getNumFreeBlocks();
        }
        return numFreeBlocksPerWindowSize;
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const
    {
        return sumWindows([](WindowBlockManager const& manager) { return manager.getNumFreeBlocks(); });
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired, SizeType32 windowSize) const
    {
        return mWindowBlockManagers.at(windowSize).schedulingHasFreeBlocks(numRequired);
    }

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumAllocTotalBlocks(); });
    }

    [[nodiscard]] SizeType32 getFirstWindowSize() const
    {
        if (mWindowBlockManagers.empty())
        {
            return 0;
        }
        return mWindowBlockManagers.begin()->first;
    }

    [[nodiscard]] SizeType32 getLastWindowSize() const
    {
        if (mWindowBlockManagers.empty())
        {
            return 0;
        }
        return mWindowBlockManagers.rbegin()->first;
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumAllocNewBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumReusedBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumMissedBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumSecondaryBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumSecondaryBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumFreeSecondaryBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumFreeSecondaryBlocks(); });
    }

    //! \brief Get per-window-size iteration stats. Resets delta snapshots for each window.
    [[nodiscard]] std::map<SizeType32, KvCacheIterationStats> getAndResetIterationStats();

    [[nodiscard]] SizeType32 getNumLayers() const
    {
        return mNumLayers;
    }

    [[nodiscard]] CacheType getCacheType() const
    {
        return mCacheType;
    }

    [[nodiscard]] SizeType32 getLayerPoolIdx(SizeType32 layerIdx) const
    {
        auto const& manager = windowManagerByLayer(layerIdx);
        auto const absoluteOffset = absolutePoolsOffset(manager);
        auto const relativePoolIndex = manager.getLayerPoolIdx(layerIdx);
        return absoluteOffset + relativePoolIndex;
    }

    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layerIdx) const
    {
        return windowManagerByLayer(layerIdx).getPoolLayerIdx(layerIdx);
    }

    [[nodiscard]] bool isPoolLayerFirst(SizeType32 layerIdx) const
    {
        auto const& manager = windowManagerByLayer(layerIdx);
        auto const relativePoolIndex = manager.getLayerPoolIdx(layerIdx);
        return manager.getPool(relativePoolIndex).layerFirstLayout;
    }

    [[nodiscard]] SizeType32 getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    [[nodiscard]] SizeType32 getStreamDevice() const
    {
        return mStream->getDevice();
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout) const;

    void flushIterationEvents()
    {
        if (mEventManager)
        {
            mEventManager->flush();
        }
    }

    [[nodiscard]] SizeType32 getPoolWindowSize(SizeType32 poolIdx) const
    {
        return mAbsolutePoolToWindowSize.at(poolIdx);
    }

    [[nodiscard]] SizeType32 getBlockSize(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).blockSize;
    }

    [[nodiscard]] SizeType32 getNumPools(
        bool includeBlockScalePools = true, bool includeIndexerKCachePools = true) const
    {
        return sumWindows([includeBlockScalePools, includeIndexerKCachePools](auto const& manager)
            { return manager.getNumPools(includeBlockScalePools, includeIndexerKCachePools); });
    }

    [[nodiscard]] std::map<SizeType32, WindowSizeMetadata> const& getWindowSizesMetadata() const noexcept
    {
        return mWindowSizeToMetadata;
    }

    [[nodiscard]] WindowSizeMetadata getWindowSizeMetadata(SizeType32 windowSize) const noexcept
    {
        return mWindowSizeToMetadata.at(windowSize);
    }

    [[nodiscard]] std::optional<LinearAttentionMetadata> const& getLinearAttentionMetadata() const noexcept
    {
        return mLinearAttentionMetadata;
    }

    [[nodiscard]] bool isVariableWindow() const noexcept
    {
        return mIsVariableWindow;
    }

    [[nodiscard]] SizeType32 getMaxBlockPerSeqWhenSingleWindowSize() const
    {
        TLLM_CHECK_WITH_INFO(!isVariableWindow(),
            "This function was called assuming there is only a single window size, and therefore a single "
            "maxBlocksPerSeq");
        auto const windowSize = windowManagerByLayer(0).getWindowSize();
        auto const onlyWindowSizeMetadata = getWindowSizeMetadata(windowSize);
        return onlyWindowSizeMetadata.maxBlocksPerSeq;
    }

    [[nodiscard]] bool isVariableGQA() const noexcept
    {
        return mIsVariableGQA;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).primaryPtr;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getSecondaryPool(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).secondaryPtr;
    }

    [[nodiscard]] SizeType32 getNumAllocatedBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumAllocatedBlocks(); });
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getMaxNumBlocks(); });
    }

    [[nodiscard]] BlockPtr getBlockById(KVCacheBlock::IdType blockId, SizeType32 windowSize) const
    {
        return mWindowBlockManagers.at(windowSize).getBlockById(blockId);
    }

    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(
        BlockKey const& blockKey, SizeType32 windowSize)
    {
        return mWindowBlockManagers.at(windowSize).findBlocksInReuseTreeByBlockKey(blockKey);
    }

    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKeys(
        std::vector<BlockKey> const& blockKeys, SizeType32 windowSize)
    {
        return mWindowBlockManagers.at(windowSize).findBlocksInReuseTreeByBlockKeys(blockKeys);
    }

    [[nodiscard]] SizeType32 getNumPrimaryBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumPrimaryBlocks(); });
    }

    [[nodiscard]] bool containsBlockScales(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).containsBlockScales;
    }

    //! \brief Store context blocks
    void storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest);

    //! \brief Store newest block for reuse
    void storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest);

    //! \brief Sync internal streams used by transfer manager with buffer manager stream
    void syncTransferManagerWithBufferManager();

    //! \brief Perform per-request bookkeeping
    void refreshBlocks();

    [[nodiscard]] WindowBlockManager& getWindowBlockManager(SizeType32 windowSize)
    {
        return mWindowBlockManagers.at(windowSize);
    }

    [[nodiscard]] runtime::BufferManager const& getBufferManager(SizeType32 windowSize) const
    {
        return mWindowBlockManagers.at(windowSize).getBufferManager();
    }

    [[nodiscard]] KVCacheBlockPool const& getPool(SizeType32 poolIdx) const
    {
        auto const windowSize = getPoolWindowSize(poolIdx);
        auto const relativePoolIndex = mAbsolutePoolToRelativePoolIndex.at(poolIdx);
        return mWindowBlockManagers.at(windowSize).getPool(relativePoolIndex);
    }

    [[nodiscard]] KVCacheBlockPool const& getRecurrentStatesPool() const
    {
        return mWindowBlockManagers.at(LinearAttentionMetadata::LinearCacheType::kRecurrentStates).getPool(0);
    }

    //! \brief Update cache offsets for blocks initiated from sequence
    void updateSequenceCacheBlockOffsets(GenerationRequest& seq, SizeType32 windowSize);

    //! \brief Update cache offsets for block at index
    void updateCacheBlockOffsetsAtIdx(GenerationRequest& seq, SizeType32 windowSize, SizeType32 blockIdx);

    //! \brief Add/detach block(s) to/from the sequence if needed
    //! \details When we need a new block, we add it. For sliding window
    //! attention (SWA), when a block goes out-of-window (OOW), we detach it
    //! If this called in the first step of the generation phase, we may
    //! detach more than a single block since there may be more than one
    //! context block that goes OOW.
    void adjustBlocksIfNeeded(GenerationRequest& sequence);

    void resetReuseState()
    {
        // Reset the shared tree once; all blocks' LookupNodePtr references to the old
        // tree are released automatically as the shared_ptrs in KVCacheBlock expire.
        mLookupTree = radix_block_tree::UnifiedBlockTree();
        for (auto& [windowSize, manager] : mWindowBlockManagers)
        {
            manager.resetReuseState();
        }
    }

private:
    [[nodiscard]] WindowBlockManager const& windowManagerByLayer(SizeType32 layerIdx) const
    {
        return mWindowBlockManagers.at(mLayerToWindowSize.at(layerIdx));
    }

    [[nodiscard]] SizeType32 sumWindows(std::function<SizeType32(WindowBlockManager const&)> produce) const
    {
        return std::accumulate(mWindowBlockManagers.cbegin(), mWindowBlockManagers.cend(), SizeType32{0},
            [&produce](SizeType32 acc, auto const& manager) { return acc + produce(manager.second); });
    }

    [[nodiscard]] SizeType32 absolutePoolsOffset(WindowBlockManager const& manager) const
    {
        auto const windowSize = manager.getWindowSize();
        return getWindowSizeMetadata(windowSize).absolutePoolsOffset;
    }

private:
    SizeType32 mNumLayers;
    SizeType32 mTokensPerBlock;
    std::shared_ptr<KVCacheEventManager> mEventManager;
    std::shared_ptr<kvc::BaseLoopbackAgent> mLoopbackAgent;
    CudaStreamPtr mStream;
    CacheType mCacheType;

    bool mIsVariableWindow;
    bool mIsVariableGQA;

    // Shared radix lookup tree used by all WindowBlockManager instances.
    // Stored before mWindowBlockManagers so it is constructed first and its address
    // is stable when passed to each WindowBlockManager constructor.
    radix_block_tree::UnifiedBlockTree mLookupTree;
    std::map<SizeType32, WindowBlockManager> mWindowBlockManagers;
    std::map<SizeType32, WindowSizeMetadata> mWindowSizeToMetadata;
    std::vector<SizeType32> mLayerToWindowSize;
    std::vector<SizeType32> mAbsolutePoolToWindowSize;
    std::vector<SizeType32> mAbsolutePoolToRelativePoolIndex;

    bool mIsEnableIndexerKCache{false};
    SizeType32 mIndexerKCacheQuantBlockSize{0};
    SizeType32 mIndexerKCacheIndexHeadDim{0};
    bool mIndexerKCacheUseFp4{false};
    std::optional<LinearAttentionMetadata> mLinearAttentionMetadata;
};

struct OffsetTableDimensions
{
    SizeType32 maxBlocksPerSeq;
    SizeType32 numPools;
    CacheType cacheType;
};

class BaseKVCacheManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    virtual ~BaseKVCacheManager() {}

    virtual void allocatePools(bool useUvm = false) = 0;

    virtual void releasePools() = 0;

    virtual void startScheduling() = 0;

    [[nodiscard]] virtual SizeType32 getTokensPerBlock() const = 0;

    [[nodiscard]] virtual SizeType32 getMaxNumBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getUsedNumBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getNumFreeBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getNumPools() const = 0;

    // only used by test
    [[nodiscard]] virtual SizeType32 getNumReusedBlocks() const noexcept = 0;

    [[nodiscard]] virtual KvCacheStats getKvCacheStats() const = 0;

    //! \brief Get per-iteration stats with delta counters, keyed by window size.
    //! Resets delta snapshots on each call.
    [[nodiscard]] virtual std::map<SizeType32, KvCacheIterationStats> getIterationStats() = 0;

    [[nodiscard]] virtual OffsetTableDimensions getOffsetTableDimensions() const = 0;

    [[nodiscard]] virtual std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const
        = 0;

    [[nodiscard]] virtual BlockManager const& getBlockManager() const = 0;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations.
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @param cachedSummary Optional pre-computed PrefixReuseSummary to avoid redundant radix tree walks.
    /// @return  The number of blocks
    [[nodiscard]] virtual SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead,
        SizeType32 windowSize, std::optional<PrefixReuseSummary> const& cachedSummary = std::nullopt) const
        = 0;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
    /// maxNewTokens).
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @param cachedSummary Optional pre-computed PrefixReuseSummary to avoid redundant radix tree walks.
    /// @return  The number of blocks
    [[nodiscard]] virtual SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req, SizeType32 windowSize,
        std::optional<PrefixReuseSummary> const& cachedSummary = std::nullopt) const
        = 0;

    /// @brief Pin blocks associated with a request to prevent eviction.
    /// @param requestId The ID of the request whose blocks should be pinned.
    virtual void pinBlocks(LlmRequest::RequestIdType requestId) = 0;

    /// @brief Increase size for request at seqSlotIdx. Allocate new KV cache block(s) if needed.
    virtual void addToken(LlmRequest::RequestIdType requestId) = 0;

    /// @brief Get the number of tokens for a request at KVCacheManager's sight. Sometimes it is different from
    /// LlmRequest::getNumTokens.
    [[nodiscard]] virtual SizeType32 getTokenCount(LlmRequest::RequestIdType requestId) const = 0;

    //! \brief Batch add sequences with two-phase claim-then-onboard strategy.
    //! \details For each attention window, when block reuse is enabled, Phase 1 claims all matching
    //!          blocks across all requests (protecting them from eviction via PartialClaimTracker),
    //!          then Phase 2 onboards host blocks and allocates non-matching blocks. When block reuse
    //!          is disabled, buildClaimResultMetadata() prepares ClaimResult metadata without radix
    //!          tree traversal, and Phase 2 performs fresh allocation only. Supports variable sliding
    //!          window attention (VSWA) by iterating over all window sizes.
    virtual void addSequenceBatch(
        std::vector<std::tuple<LlmRequest::RequestIdType, SizeType32, SizeType32>> const& requestInfos,
        std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests)
        = 0;

    [[nodiscard]] virtual std::optional<KVCacheBlock::IdType> removeSequence(LlmRequest::RequestIdType requestId,
        OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinOnRelease = false)
        = 0;

    virtual void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getBlockPoolPointers() const = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getBlockScalePoolPointers() const = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getLayerToPoolMapping() const = 0;

    virtual void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
        = 0;

    //! @return maxBlockCount of all beams
    virtual SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual bool isEnableBlockReuse() const = 0;

    [[nodiscard]] virtual bool isEnablePartialReuse() const = 0;

    [[nodiscard]] virtual bool isEnableIndexerKCache() const = 0;
    [[nodiscard]] virtual SizeType32 getIndexerKCacheIndexHeadDim() const = 0;
    [[nodiscard]] virtual SizeType32 getIndexerKCacheQuantBlockSize() const = 0;
    [[nodiscard]] virtual bool getIndexerKCacheUseFp4() const = 0;

    // void removeToken(SizeType32 seqSlotIdx);
    virtual void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) = 0;

    [[nodiscard]] virtual GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const = 0;
    [[nodiscard]] virtual GenerationRequest& getSequence(LlmRequest::RequestIdType requestId) = 0;

    [[nodiscard]] virtual bool isCrossKv() const = 0;

    //! \brief Combined prefix reuse analysis — single radix tree walk.
    //! \details Collects firstNewBlock + reusableBlocksAllocated + reusableBlocksAll in one pass.
    [[nodiscard]] virtual PrefixReuseSummary analyzePrefixReuse(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
        = 0;

    //! \brief Store full context blocks contributed by llmRequest.
    //! \details These blocks become reusable from next step.
    virtual void storeContextBlocks(LlmRequest const& llmRequest) = 0;

    //! \brief Store newest block for reuse.
    //! \details This block become reusable from next step.
    virtual void storeNewBlock(LlmRequest const& llmRequest) = 0;

    /// \brief Store blocks for reuse for a given request id
    [[nodiscard]] virtual std::vector<KVCacheBlock::IdType> storeBlocksForReuse(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks = false)
        = 0;

    //! \brief Get the block ids of a request [per beam] **for a given window size block manager**
    [[nodiscard]] virtual std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        LlmRequest::RequestIdType requestId, SizeType32 windowSize) const
        = 0;

    //! \brief Get the block ids of a batch of requests [per beam] **for a given window size block manager**
    [[nodiscard]] virtual std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const
        = 0;

    /// @brief Get the last block id (beam 0) for a given sequence and window size
    [[nodiscard]] virtual std::optional<KVCacheBlock::IdType> getLastBlockId(LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getUniquePrimaryPool() const = 0;
    [[nodiscard]] virtual runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const = 0;
    [[nodiscard]] virtual runtime::ITensor::SharedPtr getIndexerKCachePool() const = 0;
    [[nodiscard]] virtual SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const = 0;
    [[nodiscard]] virtual bool isPoolLayerFirst(SizeType32 layer_idx) const = 0;

    virtual void syncTransferManagerWithBufferManager() = 0;
    virtual void refreshBlocks() = 0;
    virtual void flushIterationEvents() = 0;
    virtual void resetReuseState() = 0;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    // Sum of numLayers * kvFactor * numKvHeads * sizePerHead for each pool
    [[nodiscard]] static SizeType32 calculateCacheSizePerTokenForSingleWindowSize(
        std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        std::vector<SizeType32> const& windowSizeLayers, SizeType32 kvFactor)
    {
        std::vector<SizeType32> nkvh;
        nkvh.reserve(windowSizeLayers.size());
        for (auto const layer : windowSizeLayers)
        {
            nkvh.push_back(numKvHeadsPerLayer.at(layer));
        }
        auto const sumLocalHeads = std::reduce(nkvh.cbegin(), nkvh.cend());
        // NOTE: We expect the caller to have already taken the tp size into account for numKvHeadsPerLayer
        // consider only local layers for the calculation
        return sumLocalHeads * kvFactor * sizePerHead;
    }

    /// @brief Groups model layers by their attention window size.
    /// @param maxAttentionWindowVec Vector of maximum attention window sizes per layer (may have fewer elements than
    /// numLayers, in which case it cycles)
    /// @param numLayers Total number of layers in the model
    /// @return Map from window size to vector of layer indices that use that window size
    [[nodiscard]] static std::map<SizeType32, std::vector<SizeType32>> groupLayersByWindowSize(
        std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 numLayers);

    /// @brief Calculate the free memory available for KV cache allocation.
    /// @param bufferManager Buffer manager for memory operations
    /// @param config KV cache configuration parameters
    /// @return Tuple containing the {.freePrimaryMemBytes, .freeSecondaryMemBytes}
    [[nodiscard]] static std::tuple<uint64_t, uint64_t> calculateFreeMemBytes(
        runtime::BufferManager const& bufferManager, executor::KvCacheConfig const& config);

    /// @brief Calculate the maximum number of KV cache blocks that can be allocated based on available GPU memory.
    /// @details This function computes how many blocks each WindowBlockManager should receive based on the weighted
    /// share
    ///          of memory requirements. The weighting considers both the window size and the number of
    ///          layers using each window size, as well as the sum of cache sizes per token for each window.
    /// @param config KV cache configuration parameters
    /// @param dtype Data type used for KV cache values
    /// @param numKvHeadsPerLayer Number of KV heads for each local layer (caller selects self/cross attention heads)
    /// @param sizePerHead Size of each attention head
    /// @param tokensPerBlock Number of tokens per KV cache block
    /// @param worldConfig World configuration for multi-GPU setups
    /// @param windowSizeToLayers Map from attention window size to vector of layer indices using that window size
    /// @param allottedPrimaryMemBytes Allotted primary memory
    /// @param allottedSecondaryMemBytes Allotted secondary memory
    /// @param extraCostMemory Additional memory cost to account for CacheTransBufferManager::preAllocBufferSize
    /// @param kvFactor Factor for KV cache size calculation (typically 2 for key+value)
    /// @param maxBatchSize Maximum batch size
    /// @return Map from window size to tuple of (primary blocks, secondary blocks)
    [[nodiscard]] static BlocksPerWindow calculateMaxNumBlocks(executor::KvCacheConfig const& config,
        nvinfer1::DataType dtype, std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, tensorrt_llm::runtime::WorldConfig const& worldConfig,
        std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers, uint64_t allottedPrimaryMemBytes,
        uint64_t allottedSecondaryMemBytes, size_t extraCostMemory, SizeType32 kvFactor, SizeType32 maxBatchSize,
        std::optional<LinearAttentionMetadata> const& linearAttentionMetadata = std::nullopt);

    /// @brief Calculates the maximum batch size that can fit the kv-cache, given that all sequences in the batch have
    /// the provided input and output length.
    ///
    /// @param inputLength The number of input tokens in each sequence in the batch.
    /// @param outputLength The number of output tokens in each sequence in the batch.
    /// @return SizeType32 A number of sequences per batch.
    [[nodiscard]] virtual SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const = 0;

    [[nodiscard]] virtual CacheType getCacheType() const = 0;

    [[nodiscard]] virtual std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(
        BlockKey const& blockKey, SizeType32 windowSize)
        = 0;

    [[nodiscard]] virtual std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKeys(
        std::vector<BlockKey> const& blockKeys, SizeType32 windowSize)
        = 0;

    virtual void unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds) = 0;

    //! @brief Get the retention priority of a block by its ID.
    //! @param blockId The ID of the block.
    //! @param windowSize The attention window size this block belongs to.
    //! @return The retention priority of the block, or default priority if block not found.
    [[nodiscard]] virtual executor::RetentionPriority getPriorityByBlockId(
        KVCacheBlock::IdType blockId, SizeType32 windowSize) const
        = 0;
};

class KVCacheManager : public BaseKVCacheManager
{
public:
    friend class KVCacheManagerBindings;

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype, SizeType32 sinkTokenLength,
        CudaStreamPtr stream, SizeType32 maxSequenceLength, SizeType32 chunkSize, bool enableBlockReuse = false,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr,
        bool enableIndexerKCache = false, SizeType32 indexerKCacheQuantBlockSize = 128,
        SizeType32 indexerKCacheIndexHeadDim = 0, bool indexerKCacheUseFp4 = false,
        std::optional<LinearAttentionMetadata> linearAttentionMetadata = std::nullopt);

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype, SizeType32 sinkTokenLength,
        int64_t stream, SizeType32 maxSequenceLength, SizeType32 chunkSize, bool enableBlockReuse = false,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr,
        bool enableIndexerKCache = false, SizeType32 indexerKCacheQuantBlockSize = 128,
        SizeType32 indexerKCacheIndexHeadDim = 0, bool indexerKCacheUseFp4 = false,
        std::optional<LinearAttentionMetadata> linearAttentionMetadata = std::nullopt);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype, SizeType32 sinkTokenLength,
        CudaStreamPtr stream, SizeType32 maxSequenceLength, SizeType32 chunkSize, bool enableBlockReuse = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr,
        bool enableIndexerKCache = false, SizeType32 indexerKCacheQuantBlockSize = 128,
        SizeType32 indexerKCacheIndexHeadDim = 0, bool indexerKCacheUseFp4 = false,
        std::optional<LinearAttentionMetadata> linearAttentionMetadata = std::nullopt);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype, SizeType32 sinkTokenLength,
        int64_t stream, SizeType32 maxSequenceLength, SizeType32 chunkSize, bool enableBlockReuse = false,
        CacheType cacheType = CacheType::kSELF, bool enablePartialReuse = true, bool copyOnpartialReuse = true,
        bool enableIndexerKCache = false, SizeType32 indexerKCacheQuantBlockSize = 128,
        SizeType32 indexerKCacheIndexHeadDim = 0, bool indexerKCacheUseFp4 = false,
        std::optional<LinearAttentionMetadata> linearAttentionMetadata = std::nullopt);

    ~KVCacheManager() override = default;

    void allocatePools(bool useUvm = false) override;

    void releasePools() override;

    void startScheduling() override;

    [[nodiscard]] SizeType32 getTokensPerBlock() const override
    {
        return mBlockManager.getTokensPerBlock();
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const override
    {
        return mBlockManager.getMaxNumBlocks();
    }

    [[nodiscard]] SizeType32 getUsedNumBlocks() const override
    {
        return mBlockManager.getNumAllocatedBlocks();
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const override
    {
        return mBlockManager.getNumFreeBlocks();
    }

    [[nodiscard]] SizeType32 getNumPools() const override
    {
        return mBlockManager.getNumPools();
    }

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mBlockManager.getNumAllocTotalBlocks();
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mBlockManager.getNumAllocNewBlocks();
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept override
    {
        return mBlockManager.getNumReusedBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mBlockManager.getNumMissedBlocks();
    }

    [[nodiscard]] std::map<SizeType32, SizeType32> getNumFreeBlocksPerWindowSize() const
    {
        return mBlockManager.getNumFreeBlocksPerWindowSize();
    }

    [[nodiscard]] KvCacheStats getKvCacheStats() const override
    {
        KvCacheStats kvCacheStats;
        kvCacheStats.maxNumBlocks = getMaxNumBlocks();
        kvCacheStats.freeNumBlocks = getNumFreeBlocks();
        kvCacheStats.usedNumBlocks = getUsedNumBlocks();
        kvCacheStats.toksPerBlock = getTokensPerBlock();
        kvCacheStats.allocTotalBlocks = getNumAllocTotalBlocks();
        kvCacheStats.allocNewBlocks = getNumAllocNewBlocks();
        kvCacheStats.reusedBlocks = getNumReusedBlocks();
        kvCacheStats.missedBlocks = getNumMissedBlocks();
        kvCacheStats.cacheHitRate = kvCacheStats.reusedBlocks == 0 ? 0
                                                                   : static_cast<float>(kvCacheStats.reusedBlocks)
                / static_cast<float>(kvCacheStats.reusedBlocks + kvCacheStats.missedBlocks);
        kvCacheStats.numFreeBlocksPerWindowSize = getNumFreeBlocksPerWindowSize();
        kvCacheStats.allocatedBytes = mAllocatedBytes;
        return kvCacheStats;
    }

    [[nodiscard]] std::map<SizeType32, KvCacheIterationStats> getIterationStats() override
    {
        return mBlockManager.getAndResetIterationStats();
    }

    [[nodiscard]] OffsetTableDimensions getOffsetTableDimensions() const override
    {
        OffsetTableDimensions dims;
        // We use the mMaxAttentionWindow here, because we prefer to have a single offset table for simplicity,
        // And we don't mind that it should be as wide as the widest window, because that is negligible.
        dims.maxBlocksPerSeq = mBlockManager.getWindowSizeMetadata(mMaxAttentionWindow).maxBlocksPerSeq;
        dims.numPools = mBlockManager.getNumPools();
        dims.cacheType = mBlockManager.getCacheType();
        return dims;
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        return mBlockManager.getLatestEvents(timeout);
    }

    [[nodiscard]] BlockManager const& getBlockManager() const override
    {
        return mBlockManager;
    }

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead,
        SizeType32 windowSize, std::optional<PrefixReuseSummary> const& cachedSummary = std::nullopt) const override;

    /// @brief  Function that computes the number of KV cache blocks remaining to advance a request to completion (i.e.
    /// for maxNewTokens); the allocated blocks are excluded
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req, SizeType32 windowSize,
        std::optional<PrefixReuseSummary> const& cachedSummary = std::nullopt) const override;

    /// @brief Increase size for request with requestId. Allocate new KV cache block(s) if needed.
    void addToken(LlmRequest::RequestIdType requestId) override;

    /// @brief LlmRequest::getNumTokens is out of sync with GenerationRequest when overlap scheduler is enabled.
    /// This function returns the correct number of tokens from GenerationRequest to keep the behavior consistent.
    [[nodiscard]] SizeType32 getTokenCount(LlmRequest::RequestIdType requestId) const override;

    //! \brief According to request's current position, copy data from the last full block to the next block (ignoring
    //! the placeholder block). It should be called before every forward step, after adding new tokens.
    void copyLinearAttentionBlock(LlmRequest const& llmRequest);

    void addSequenceBatch(
        std::vector<std::tuple<LlmRequest::RequestIdType, SizeType32, SizeType32>> const& requestInfos,
        std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests) override;

    [[nodiscard]] std::optional<KVCacheBlock::IdType> removeSequence(LlmRequest::RequestIdType requestId,
        OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinOnRelease = false) override;

    void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) override;

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        return mBlockPoolPointers;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        return mLayerToPoolMapping;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockScalePoolPointers() const override
    {
        // TODO: add a new optional model input so the attention plugin can access these
        return mBlockScalePoolPointers;
    }

    void getBlockOffsetsOfBatch(runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
        SizeType32 beamWidth) const override;

    //! @return maxBlockCount of all beams
    SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const override;

    [[nodiscard]] bool isEnableBlockReuse() const override
    {
        return mEnableBlockReuse;
    }

    [[nodiscard]] bool isEnablePartialReuse() const override
    {
        return mBlockManager.isEnablePartialReuse();
    }

    [[nodiscard]] bool isEnableIndexerKCache() const override
    {
        return mBlockManager.isEnableIndexerKCache();
    }

    [[nodiscard]] SizeType32 getIndexerKCacheIndexHeadDim() const override
    {
        return mBlockManager.getIndexerKCacheIndexHeadDim();
    }

    [[nodiscard]] SizeType32 getIndexerKCacheQuantBlockSize() const override
    {
        return mBlockManager.getIndexerKCacheQuantBlockSize();
    }

    [[nodiscard]] bool getIndexerKCacheUseFp4() const override
    {
        return mBlockManager.getIndexerKCacheUseFp4();
    }

    void removeToken(LlmRequest::RequestIdType requestId);
    void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override;

    [[nodiscard]] GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const override;
    [[nodiscard]] GenerationRequest& getSequence(LlmRequest::RequestIdType requestId) override;

    [[nodiscard]] bool isCrossKv() const override
    {
        return mBlockManager.getCacheType() == CacheType::kCROSS;
    }

    [[nodiscard]] CacheType getCacheType() const override
    {
        return mBlockManager.getCacheType();
    }

    //! \brief Combined prefix reuse analysis — single radix tree walk.
    [[nodiscard]] PrefixReuseSummary analyzePrefixReuse(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const override;

    //! \brief Store full context blocks contributed by llmRequest.
    //! \details These blocks become reusable from next step.
    void storeContextBlocks(LlmRequest const& llmRequest) override;

    //! \brief Store newest blocks for reuse
    void storeNewBlock(LlmRequest const& llmRequest) override;

    [[nodiscard]] std::vector<KVCacheBlock::IdType> storeBlocksForReuse(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks = false) override;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override;

    /// @brief Calculates the number of kv-cache blocks that a sequence will require.
    ///
    /// @param inputLength The number of input tokens in the sequence.
    /// @param outputLength The number of output tokens in the sequence.
    /// @param sinkTokenLength The number of sink tokens configured.
    /// @param maxAttentionWindow The attention window size allowed by the model.
    /// @param beamWidth The number of beams to consider for the request.
    /// @param tokensPerBlock The number of tokens a single kv-cache block contains.,
    /// @return SizeType32 A number of blocks.
    [[nodiscard]] static SizeType32 calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 beamWidth, SizeType32 tokensPerBlock);

    void pinBlocks(LlmRequest::RequestIdType requestId) override;

    void unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds) override;

    [[nodiscard]] executor::RetentionPriority getPriorityByBlockId(
        KVCacheBlock::IdType blockId, SizeType32 windowSize) const override;

    std::optional<KVCacheBlock::IdType> getLastBlockId(LlmRequest::RequestIdType requestId) const override;

    /// @brief Calculates the number of kv-cache blocks that a sequence will require, for a single beam.
    ///
    /// @param sequenceLength The total length of the sequence (input and output).
    /// @param sinkTokenLength The number of sink tokens configured.
    /// @param windowSize The attention window size
    /// @param tokensPerBlock The number of tokens in a single kv-cache block.
    /// @return SizeType32 A number of blocks.
    [[nodiscard]] static SizeType32 calculateMaxBlockRequirementsPerBeam(
        SizeType32 sequenceLength, SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 tokensPerBlock);

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        LlmRequest::RequestIdType requestId, SizeType32 windowSize) const override;

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const override;

    runtime::ITensor::SharedPtr getUniquePrimaryPool() const override;
    runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override;
    runtime::ITensor::SharedPtr getIndexerKCachePool() const override;

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        return mBlockManager.getPoolLayerIdx(layer_idx);
    }

    bool isPoolLayerFirst(SizeType32 layer_idx) const override
    {
        return mBlockManager.isPoolLayerFirst(layer_idx);
    }

    void syncTransferManagerWithBufferManager() override
    {
        mBlockManager.syncTransferManagerWithBufferManager();
    }

    //! \brief Perform per-iteration bookkeeping
    void refreshBlocks() override
    {
        mBlockManager.refreshBlocks();
    }

    void flushIterationEvents() override
    {
        mBlockManager.flushIterationEvents();
    }

    std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(
        BlockKey const& blockKey, SizeType32 windowSize) override
    {
        return mBlockManager.findBlocksInReuseTreeByBlockKey(blockKey, windowSize);
    }

    std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKeys(
        std::vector<BlockKey> const& blockKeys, SizeType32 windowSize) override
    {
        return mBlockManager.findBlocksInReuseTreeByBlockKeys(blockKeys, windowSize);
    }

    void resetReuseState() override
    {
        mBlockManager.resetReuseState();
    }

    /// @brief Finds the maximum attention window that can be used on a sequence, given some kv-cache block capacity.
    ///
    /// @param inputLength The number of input tokens in the sequence.
    /// @param outputLength The number of output tokens in the sequence.
    /// @param sinkTokenLength The number of sink tokens.
    /// @param blockCapacity The number of kv-cache blocks available.
    /// @param beamWidth The number of beams to consider.
    /// @param tokensPerBlock The number of tokens per kv-cache block.
    /// @return SizeType32 A maximum attention window in number of tokens.
    [[nodiscard]] static SizeType32 calculateMaxAttentionWindow(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 blockCapacity, SizeType32 beamWidth, SizeType32 tokensPerBlock);

private:
    // Maximum number of sequences
    SizeType32 mMaxNumSequences;
    // Maximum beam width
    SizeType32 mMaxBeamWidth;
    nvinfer1::DataType mDataType;
    // Maximum kv cache length per sequence
    SizeType32 mMaxAttentionWindow;
    // Number of tokens per block
    SizeType32 mTokensPerBlock;
    // Number of tokens to fill up the sink tokens to a full block size
    SizeType32 mSinkBubbleLength;
    // Number of tokens in the sink blocks
    SizeType32 mSinkBlockTokenLength;
    // Number of tokens in a chunk. If chunked-prefill is not enabled, this will be the same as max sequence length.
    SizeType32 mChunkSize;
    // Block manager
    BlockManager mBlockManager;
    // Map of all sequences
    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;
    // Whether to cache KV pages for reuse
    bool mEnableBlockReuse;
    // Mutex to protect access to mSequences
    mutable std::mutex mSequencesMtx;
    // buffers for static tensors, will be created after allocating pools
    runtime::ITensor::SharedPtr mBlockPoolPointers;
    runtime::ITensor::SharedPtr mLayerToPoolMapping;
    runtime::ITensor::SharedPtr mBlockScalePoolPointers;
    runtime::ITensor::SharedPtr mIndexerKCachePoolPointers;
    // GPU bytes allocated for KV-cache
    std::size_t mAllocatedBytes{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
""",
    'key-files/cpp/include/tensorrt_llm/executor/dataTransceiverState.h': r"""/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tensorrt_llm::executor
{

class Serialization;

namespace kv_cache
{

// Describe the data structure for cache layout, which can be used to infer
// cache layouts and location associations between different processes, in order
// to determine suitable senders and receivers.
class CacheState final
{
public:
    struct ModelConfig;

    enum class AttentionType : std::uint8_t
    {
        kDEFAULT = 0,
        kMLA = 1,
    };

    CacheState(ModelConfig modelConfig, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        AttentionType attentionType = AttentionType::kDEFAULT, int kvFactor = 2, bool enableBlockReuse = false,
        bool enablePartialReuse = false, bool hasIndexerKCache = false, SizeType32 indexerDimPerHead = 0,
        SizeType32 indexerKCacheQuantBlockSize = 128, bool indexerKCacheUseFp4 = false)
        : mModelConfig(std::move(modelConfig))
        , mParallelConfig{worldConfig.getTensorParallelism(), worldConfig.getPipelineParallelism(),
              worldConfig.getContextParallelism(), worldConfig.enableAttentionDP(), worldConfig.getTensorParallelRank(),
              worldConfig.getTensorParallelism(), attentionLayerNumPerPP}
        , mDataType{dataType}
        , mAttentionConfig(attentionType, kvFactor)
    {
        mEnableBlockReuse = enableBlockReuse;
        mEnablePartialReuse = enablePartialReuse;
        mHasIndexerKCache = hasIndexerKCache;
        mIndexerDimPerHead = indexerDimPerHead;
        mIndexerKCacheQuantBlockSize = indexerKCacheQuantBlockSize;
        mIndexerKCacheUseFp4 = indexerKCacheUseFp4;
    }

    CacheState(std::vector<SizeType32> nbKvHeadPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, SizeType32 contextParallelism,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        AttentionType attentionType = AttentionType::kDEFAULT, int kvFactor = 2, bool enableAttentionDP = false,
        int DPrank = 0, int DPsize = 0, bool enableBlockReuse = false, bool enablePartialReuse = false,
        bool hasIndexerKCache = false, SizeType32 indexerDimPerHead = 0, SizeType32 indexerKCacheQuantBlockSize = 128,
        bool indexerKCacheUseFp4 = false)
        : mModelConfig{std::move(nbKvHeadPerLayer), sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism, contextParallelism, enableAttentionDP, DPrank, DPsize,
              attentionLayerNumPerPP}
        , mDataType{dataType}
        , mAttentionConfig(attentionType, kvFactor)
    {
        mEnableBlockReuse = enableBlockReuse;
        mEnablePartialReuse = enablePartialReuse;
        mHasIndexerKCache = hasIndexerKCache;
        mIndexerDimPerHead = indexerDimPerHead;
        mIndexerKCacheQuantBlockSize = indexerKCacheQuantBlockSize;
        mIndexerKCacheUseFp4 = indexerKCacheUseFp4;
    }

    CacheState(SizeType32 nbAttentionLayers, SizeType32 nbKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, SizeType32 contextParallelism,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        AttentionType attentionType = AttentionType::kDEFAULT, int kvFactor = 2, bool enableAttentionDP = false,
        int DPrank = 0, int DPsize = 0, bool enableBlockReuse = false, bool enablePartialReuse = false,
        bool hasIndexerKCache = false, SizeType32 indexerDimPerHead = 0, SizeType32 indexerKCacheQuantBlockSize = 128,
        bool indexerKCacheUseFp4 = false)
        : mModelConfig{std::vector(nbAttentionLayers, nbKvHeads), sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism, contextParallelism, enableAttentionDP, DPrank, DPsize,
              attentionLayerNumPerPP}
        , mDataType{dataType}
        , mAttentionConfig(attentionType, kvFactor)
    {
        mEnableBlockReuse = enableBlockReuse;
        mEnablePartialReuse = enablePartialReuse;
        mHasIndexerKCache = hasIndexerKCache;
        mIndexerDimPerHead = indexerDimPerHead;
        mIndexerKCacheQuantBlockSize = indexerKCacheQuantBlockSize;
        mIndexerKCacheUseFp4 = indexerKCacheUseFp4;
    }

    [[nodiscard]] bool operator==(kv_cache::CacheState const& other) const noexcept
    {
        return mModelConfig == other.mModelConfig && mParallelConfig == other.mParallelConfig
            && mAttentionConfig == other.mAttentionConfig && mDataType == other.mDataType
            && mRnnCacheState == other.mRnnCacheState && mEnableBlockReuse == other.mEnableBlockReuse
            && mEnablePartialReuse == other.mEnablePartialReuse && mHasIndexerKCache == other.mHasIndexerKCache
            && mIndexerDimPerHead == other.mIndexerDimPerHead
            && mIndexerKCacheQuantBlockSize == other.mIndexerKCacheQuantBlockSize
            && mIndexerKCacheUseFp4 == other.mIndexerKCacheUseFp4;
    }

    struct ModelConfig
    {
        std::vector<SizeType32> mNbKvHeadsPerLayer;
        SizeType32 mSizePerHead;
        SizeType32 mTokensPerBlock;

        [[nodiscard]] bool operator==(ModelConfig const& other) const noexcept
        {
            return mNbKvHeadsPerLayer == other.mNbKvHeadsPerLayer && mSizePerHead == other.mSizePerHead
                && mTokensPerBlock == other.mTokensPerBlock;
        }
    };

    struct ParallelConfig
    {
        SizeType32 mTensorParallelism;
        SizeType32 mPipelineParallelism;
        SizeType32 mContextParallelism;
        bool mEnableAttentionDP;
        SizeType32 mDPrank;
        SizeType32 mDPsize;
        // number of attention layers per pipeline parallelism rank, the size of the vector is equal to the pipeline
        // parallelism size.
        std::vector<SizeType32> mAttentionLayerNumPerPP;

        [[nodiscard]] bool operator==(ParallelConfig const& other) const noexcept
        {
            return mTensorParallelism == other.mTensorParallelism && mPipelineParallelism == other.mPipelineParallelism
                && mContextParallelism == other.mContextParallelism && mEnableAttentionDP == other.mEnableAttentionDP
                && mDPrank == other.mDPrank && mDPsize == other.mDPsize
                && mAttentionLayerNumPerPP == other.mAttentionLayerNumPerPP;
        }
    };

    struct AttentionConfig
    {

        AttentionConfig(AttentionType attentionType, int kvFactor)
            : mAttentionType(attentionType)
            , mKvFactor(kvFactor)

        {
        }

        [[nodiscard]] bool operator==(AttentionConfig const& other) const noexcept
        {
            return mAttentionType == other.mAttentionType && mKvFactor == other.mKvFactor;
        }

        // attentionType ;
        AttentionType mAttentionType;
        int mKvFactor;
    };

    struct RnnModelConfig
    {
        SizeType32 mDState;      // SSM state dimension
        SizeType32 mDConv;       // Conv state dimension (convKernel - 1)
        SizeType32 mHiddenSize;  // Hidden dimension
        SizeType32 mHeadDim;     // Head dimension (0 for Mamba1, >0 for Mamba2)
        SizeType32 mConvDimSize; // Conv dimension size
        SizeType32 mNGroups;     // Number of groups (for Mamba2)
        SizeType32 mNumLayers;   // Number of layers
        SizeType32 mNumHeads;    // Number of heads

        [[nodiscard]] bool operator==(RnnModelConfig const& other) const noexcept
        {
            return mDState == other.mDState && mDConv == other.mDConv && mHiddenSize == other.mHiddenSize
                && mHeadDim == other.mHeadDim && mConvDimSize == other.mConvDimSize && mNGroups == other.mNGroups
                && mNumLayers == other.mNumLayers && mNumHeads == other.mNumHeads;
        }
    };

    /// @brief Groups all RNN/Mamba cache configuration within CacheState.
    struct RnnCacheState
    {
        RnnModelConfig mModelConfig;
        /// Number of RNN layers per pipeline parallelism rank.
        std::vector<SizeType32> mLayerNumPerPP;
        nvinfer1::DataType mConvStateDataType;
        nvinfer1::DataType mSsmStateDataType;

        [[nodiscard]] bool operator==(RnnCacheState const& other) const noexcept
        {
            return mModelConfig == other.mModelConfig && mLayerNumPerPP == other.mLayerNumPerPP
                && mConvStateDataType == other.mConvStateDataType && mSsmStateDataType == other.mSsmStateDataType;
        }
    };

    [[nodiscard]] ModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] ParallelConfig const& getParallelConfig() const
    {
        return mParallelConfig;
    }

    [[nodiscard]] AttentionConfig const& getAttentionConfig() const
    {
        return mAttentionConfig;
    }

    [[nodiscard]] nvinfer1::DataType const& getDataType() const
    {
        return mDataType;
    }

    [[nodiscard]] bool getEnableBlockReuse() const
    {
        return mEnableBlockReuse;
    }

    [[nodiscard]] bool getEnablePartialReuse() const
    {
        return mEnablePartialReuse;
    }

    [[nodiscard]] bool getHasIndexerKCache() const
    {
        return mHasIndexerKCache;
    }

    [[nodiscard]] SizeType32 getIndexerDimPerHead() const
    {
        return mIndexerDimPerHead;
    }

    [[nodiscard]] SizeType32 getIndexerKCacheQuantBlockSize() const
    {
        return mIndexerKCacheQuantBlockSize;
    }

    [[nodiscard]] bool getIndexerKCacheUseFp4() const
    {
        return mIndexerKCacheUseFp4;
    }

    // =========================================================================
    // RNN/Mamba cache state (optional, present only for hybrid models)
    // =========================================================================

    [[nodiscard]] bool hasRnnConfig() const noexcept
    {
        return mRnnCacheState.has_value();
    }

    void setRnnConfig(RnnModelConfig rnnModelConfig, std::vector<SizeType32> rnnLayerNumPerPP,
        nvinfer1::DataType convStateDataType, nvinfer1::DataType ssmStateDataType)
    {
        mRnnCacheState = RnnCacheState{
            std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType};
    }

    [[nodiscard]] RnnCacheState const& getRnnCacheState() const
    {
        TLLM_CHECK(mRnnCacheState.has_value());
        return mRnnCacheState.value();
    }

    [[nodiscard]] RnnModelConfig const& getRnnModelConfig() const
    {
        return getRnnCacheState().mModelConfig;
    }

    [[nodiscard]] nvinfer1::DataType getConvStateDataType() const
    {
        return getRnnCacheState().mConvStateDataType;
    }

    [[nodiscard]] nvinfer1::DataType getSsmStateDataType() const
    {
        return getRnnCacheState().mSsmStateDataType;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        sstring << "numKvHeads\n";
        for (auto kvHead : mModelConfig.mNbKvHeadsPerLayer)
        {
            sstring << kvHead << "\n";
        }
        sstring << "sizePerHead:" << mModelConfig.mSizePerHead << "\n";
        sstring << "mTokensPerBlock:" << mModelConfig.mTokensPerBlock << "\n";
        sstring << "tp:" << mParallelConfig.mTensorParallelism << "\n";
        sstring << "pp:" << mParallelConfig.mPipelineParallelism << "\n";
        sstring << "cp:" << mParallelConfig.mContextParallelism << "\n";
        sstring << "enableAttentionDP:" << mParallelConfig.mEnableAttentionDP << "\n";
        sstring << "datatype:" << static_cast<int32_t>(mDataType) << "\n";
        sstring << "attentionType:" << static_cast<int32_t>(mAttentionConfig.mAttentionType) << "\n";
        sstring << "kvFactor:" << mAttentionConfig.mKvFactor << "\n";
        sstring << "dpRank:" << mParallelConfig.mDPrank << "\n";
        sstring << "dpSize:" << mParallelConfig.mDPsize << "\n";
        sstring << "enableBlockReuse:" << mEnableBlockReuse << "\n";
        sstring << "enablePartialReuse:" << mEnablePartialReuse << "\n";
        sstring << "hasIndexerKCache:" << mHasIndexerKCache << "\n";
        sstring << "indexerDimPerHead:" << mIndexerDimPerHead << "\n";
        sstring << "indexerKCacheQuantBlockSize:" << mIndexerKCacheQuantBlockSize << "\n";
        sstring << "indexerKCacheUseFp4:" << mIndexerKCacheUseFp4 << "\n";
        if (mRnnCacheState.has_value())
        {
            auto const& rnn = mRnnCacheState.value();
            sstring << "RnnCacheState:\n";
            sstring << "  dState:" << rnn.mModelConfig.mDState << "\n";
            sstring << "  dConv:" << rnn.mModelConfig.mDConv << "\n";
            sstring << "  hiddenSize:" << rnn.mModelConfig.mHiddenSize << "\n";
            sstring << "  headDim:" << rnn.mModelConfig.mHeadDim << "\n";
            sstring << "  convDimSize:" << rnn.mModelConfig.mConvDimSize << "\n";
            sstring << "  nGroups:" << rnn.mModelConfig.mNGroups << "\n";
            sstring << "  numLayers:" << rnn.mModelConfig.mNumLayers << "\n";
            sstring << "  numHeads:" << rnn.mModelConfig.mNumHeads << "\n";
            sstring << "  convStateDataType:" << static_cast<int32_t>(rnn.mConvStateDataType) << "\n";
            sstring << "  ssmStateDataType:" << static_cast<int32_t>(rnn.mSsmStateDataType) << "\n";
        }
        return sstring.str();
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    ModelConfig mModelConfig;
    ParallelConfig mParallelConfig;
    nvinfer1::DataType mDataType;
    AttentionConfig mAttentionConfig;
    bool mEnableBlockReuse{false};
    bool mEnablePartialReuse{false};
    bool mHasIndexerKCache{false};
    SizeType32 mIndexerDimPerHead{0};
    SizeType32 mIndexerKCacheQuantBlockSize{128};
    bool mIndexerKCacheUseFp4{false};
    // RNN/Mamba cache state (optional, for hybrid models)
    std::optional<RnnCacheState> mRnnCacheState;
};

struct MpiState
{
    [[nodiscard]] bool operator==(MpiState const& other) const noexcept
    {
        return mRanks == other.mRanks;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        sstring << "[";
        for (auto rank : mRanks)
        {
            sstring << rank << ",";
        }
        sstring << "]";
        return sstring.str();
    }

    std::vector<SizeType32> mRanks;
};

struct SocketState
{
    [[nodiscard]] bool operator==(SocketState const& other) const noexcept
    {
        return mPort == other.mPort && mIp == other.mIp;
    }

    [[nodiscard]] std::string toString() const
    {
        return mIp + ":" + std::to_string(mPort);
    }

    std::uint16_t mPort;
    std::string mIp;
};

struct AgentState
{
    AgentState(std::string agentName, std::string connectionInfo)
        : mAgentName(std::move(agentName))
        , mConnectionInfo(std::move(connectionInfo))
    {
    }

    AgentState() = default;

    [[nodiscard]] bool operator==(AgentState const& other) const noexcept
    {
        return mAgentName == other.mAgentName && mConnectionInfo == other.mConnectionInfo;
    }

    [[nodiscard]] std::string toString() const
    {
        return mAgentName;
    }

    std::string mAgentName;
    std::string mConnectionInfo;
};

class CommState final
{
public:
    CommState() = default;

    explicit CommState(std::vector<SizeType32> ranks, int selfIdx = -1)
        : mState{MpiState{std::move(ranks)}}
        , mSelfIdx{selfIdx}
    {
    }

    explicit CommState(std::vector<SocketState> socketState, int selfIdx = -1)
        : mState{std::move(socketState)}
        , mSelfIdx{selfIdx}
    {
    }

    CommState(std::uint16_t port, std::string ip)
        : mState{std::vector<SocketState>{SocketState{port, std::move(ip)}}}
        , mSelfIdx{0}
    {
    }

    explicit CommState(std::vector<AgentState> agentState, int selfIdx = -1)
        : mState{std::move(agentState)}
        , mSelfIdx{selfIdx}
    {
    }

    [[nodiscard]] bool isMpiState() const noexcept
    {
        return std::holds_alternative<MpiState>(mState);
    }

    [[nodiscard]] bool isSocketState() const noexcept
    {
        return std::holds_alternative<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] bool isAgentState() const noexcept
    {
        return std::holds_alternative<std::vector<AgentState>>(mState);
    }

    [[nodiscard]] MpiState const& getMpiState() const
    {
        TLLM_CHECK(isMpiState());
        return std::get<MpiState>(mState);
    }

    [[nodiscard]] std::vector<SocketState> const& getSocketState() const
    {
        TLLM_CHECK(isSocketState());
        return std::get<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] std::vector<AgentState> const& getAgentState() const
    {
        TLLM_CHECK(isAgentState());
        return std::get<std::vector<AgentState>>(mState);
    }

    [[nodiscard]] int getSelfIdx() const noexcept
    {
        return mSelfIdx;
    }

    [[nodiscard]] bool operator==(CommState const& other) const noexcept
    {
        return mState == other.mState;
    }

    [[nodiscard]] std::string toString() const
    {
        if (isMpiState())
        {
            return "MPI:" + getMpiState().toString();
        }
        if (isSocketState())
        {
            std::stringstream sstring;
            sstring << "SOCKET:[";
            for (auto&& socket : getSocketState())
            {
                sstring << socket.toString() << ",";
            }
            sstring << "]";
            return sstring.str();
        }
        if (isAgentState())
        {
            std::stringstream sstring;
            sstring << "AGENT:[";
            for (auto&& agent : getAgentState())
            {
                sstring << agent.toString() << ",";
            }
            sstring << "]";
            return sstring.str();
        }

        return "";
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    std::variant<std::monostate, MpiState, std::vector<SocketState>, std::vector<AgentState>> mState;
    int mSelfIdx{-1};
};

} // namespace kv_cache

class DataTransceiverState final
{
public:
    DataTransceiverState() = default;

    DataTransceiverState(kv_cache::CacheState cacheState, kv_cache::CommState commState)
        : mCacheState{std::move(cacheState)}
        , mCommState{std::move(commState)}
    {
    }

    void setCacheState(kv_cache::CacheState state)
    {
        mCacheState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CacheState> const& getCacheState() const noexcept
    {
        return mCacheState;
    }

    void setCommState(kv_cache::CommState state)
    {
        mCommState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CommState> const& getCommState() const noexcept
    {
        return mCommState;
    }

    /// @brief Check if the cache state has RNN configuration.
    [[nodiscard]] bool hasRnnCacheState() const noexcept
    {
        return mCacheState.has_value() && mCacheState->hasRnnConfig();
    }

    [[nodiscard]] bool operator==(DataTransceiverState const& other) const noexcept
    {
        return mCacheState == other.mCacheState && mCommState == other.mCommState;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        if (mCacheState)
        {
            sstring << mCacheState.value().toString();
        }
        if (mCommState)
        {
            sstring << mCommState.value().toString();
        }
        return sstring.str();
    }

private:
    friend class Serialization;
    std::optional<kv_cache::CacheState> mCacheState;
    std::optional<kv_cache::CommState> mCommState;
};

} // namespace tensorrt_llm::executor
""",
    'key-files/cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp': r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/types.h"
#include <cstdint>
#include <limits>
#include <sstream>
#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"
#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheType.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/mlaCacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

std::mutex CacheTransceiver::mDllMutex;

std::unique_ptr<BaseCacheTransceiver> CacheTransceiverFactory::createCacheTransceiver(
    kv_cache_manager::BaseKVCacheManager* cacheManager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig)
{
    if (!cacheTransceiverConfig.has_value() || !cacheTransceiverConfig.value().getBackendType().has_value())
    {
        TLLM_LOG_INFO("CacheTransceiver is disabled.");
        return nullptr;
    }
    auto backendType = cacheTransceiverConfig.value().getBackendType();
    if (backendType.value() == executor::CacheTransceiverConfig::BackendType::DEFAULT)
    {
        if (common::getEnvUseUCXKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::UCX;
            TLLM_LOG_INFO("Enable UCX KV cache transport.");
        }
        else if (common::getEnvUseNixlKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::NIXL;
            TLLM_LOG_INFO("Enable NIXL KV cache transport.");
        }
        else if (common::getEnvUseMooncakeKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::MOONCAKE;
            TLLM_LOG_INFO("Enable MOONCAKE KV cache transport.");
        }
        else if (common::getEnvUseMPIKvCache())
        {
            backendType = executor::CacheTransceiverConfig::BackendType::MPI;
            TLLM_LOG_INFO("Enable MPI KV cache transport.");
            TLLM_LOG_WARNING("MPI KV cache transport is deprecated, please use UCX or NIXL instead.");
        }
        else
        {
            backendType = executor::CacheTransceiverConfig::BackendType::NIXL;
        }
    }
    cacheTransceiverConfig.value().setBackendType(backendType);

    executor::kv_cache::CacheState::ModelConfig cacheStateCfg{
        modelConfig.getNumKvHeadsPerLayer(), modelConfig.getSizePerHead(), modelConfig.getTokensPerBlock()};

    auto ppSize = worldConfig.getPipelineParallelism();

    std::vector<SizeType32> attentionLayerNumPerPP(ppSize, 0);
    for (int ppRank = 0; ppRank < ppSize; ppRank++)
    {
        attentionLayerNumPerPP[ppRank] = modelConfig.getNbAttentionLayers(ppSize, ppRank);
    }

    return std::make_unique<CacheTransceiver>(cacheManager, cacheStateCfg, worldConfig, attentionLayerNumPerPP,
        modelConfig.getKvDataType(), attentionType, cacheTransceiverConfig);
}

CacheTransceiver::CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
    executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
    std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
    executor::kv_cache::CacheState::AttentionType attentionType,
    std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig,
    rnn_state_manager::RnnStateManager* rnnStateManager, std::vector<SizeType32> const& rnnLayerNumPerPP)
    : mCacheTransceiverConfig{cacheTransceiverConfig}
    , mRnnStateManager{rnnStateManager}
{
    using tensorrt_llm::batch_manager::kv_cache_manager::CacheFormatter;
    if (useMPI())
    {
        mGroupComm = std::make_shared<CacheTransceiverComm>(std::addressof(tensorrt_llm::mpi::MpiComm::session()));
    }
    else
    {
        mGroupComm = std::make_shared<CacheTransceiverComm>(tensorrt_llm::pg_utils::get_world_pg());
    }

    if (worldConfig.isTensorParallel() || worldConfig.isContextParallel())
    {
        mGroupTensorParaComm = std::make_shared<CacheTransceiverComm>(
            mGroupComm->split(worldConfig.getPipelineParallelRank(), worldConfig.getRank()));
    }
    int kvFactor = 2;
    if (cacheManager->getCacheType() == kv_cache_manager::CacheType::kSELFKONLY)
    {
        kvFactor = 1;
    }
    mCacheState
        = std::make_unique<executor::kv_cache::CacheState>(cacheStateModelCfg, worldConfig, attentionLayerNumPerPP,
            dataType, attentionType, kvFactor, cacheManager->isEnableBlockReuse(), cacheManager->isEnablePartialReuse(),
            cacheManager->isEnableIndexerKCache(), cacheManager->getIndexerKCacheIndexHeadDim(),
            cacheManager->getIndexerKCacheQuantBlockSize(), cacheManager->getIndexerKCacheUseFp4());

    if (mCacheState->getParallelConfig().mEnableAttentionDP)
    {
        int dpSize = mCacheState->getParallelConfig().mDPsize;

        // dpRank is derived from the tensor parallel rank, which already accounts for CP.
        // Layout: rank = ppRank * (TP * CP) + tpRank * CP + cpRank.
        // getTensorParallelRank() correctly extracts tpRank regardless of CP.
        int dpRank = mCacheState->getParallelConfig().mDPrank;
        // <PP,DP,TP,CP>
        mGroupDataComm = std::make_shared<CacheTransceiverComm>(mGroupComm->split(dpRank, worldConfig.getRank()));
        if (worldConfig.isTensorParallel() || worldConfig.isContextParallel())
        {
            // Group ranks with same (ppRank, dpRank) accounting for CP.
            mGroupTPInDPComm = std::make_shared<CacheTransceiverComm>(
                mGroupComm->split(worldConfig.getPipelineParallelRank() * dpSize + dpRank, worldConfig.getRank()));
        }
    }
    bool isMLA = attentionType == executor::kv_cache::CacheState::AttentionType::kMLA;
    TLLM_CHECK_WITH_INFO(mCacheTransceiverConfig.has_value(), "CacheTransceiverConfig is not set.");
    auto backendType = mCacheTransceiverConfig.value().getBackendType();
    TLLM_CHECK_WITH_INFO(
        backendType.has_value() && (backendType.value() != executor::CacheTransceiverConfig::BackendType::DEFAULT),
        " CacheTransceiverConfig::BackendType is not set.");

    std::optional<size_t> maxNumTokens = mCacheTransceiverConfig.value().getMaxTokensInBuffer();

    mCacheTransBufferManagers.push_back(
        std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens));
    if (isMLA && cacheManager->isEnableIndexerKCache())
    {
        mCacheTransBufferManagers.push_back(
            std::make_unique<kv_cache_manager::CacheTransBufferManager>(cacheManager, maxNumTokens, true));
    }

    // RNN specific setup
    if (mRnnStateManager != nullptr)
    {
        TLLM_LOG_DEBUG("Setting up RNN cache transfer components.");
        TLLM_CHECK(!rnnLayerNumPerPP.empty());

        mRnnCacheTransBufferManager
            = std::make_unique<rnn_state_manager::RnnCacheTransBufferManager>(mRnnStateManager, maxNumTokens);

        auto rnnModelCfg = mRnnStateManager->getRnnCacheStateModelConfig();

        auto const convStateDataType = mRnnStateManager->getConvStateDataType();
        auto const ssmStateDataType = mRnnStateManager->getSsmStateDataType();

        mCacheState->setRnnConfig(rnnModelCfg, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);

        TLLM_LOG_INFO("RNN cache transfer components initialized.");
    }

    mCacheTransBufferManagerPtrs.clear();
    mCacheTransBufferManagerPtrs.reserve(mCacheTransBufferManagers.size() + (mRnnCacheTransBufferManager ? 1 : 0));
    for (auto& manager : mCacheTransBufferManagers)
    {
        mCacheTransBufferManagerPtrs.push_back(manager.get());
    }
    if (mRnnCacheTransBufferManager)
    {
        mCacheTransBufferManagerPtrs.push_back(mRnnCacheTransBufferManager.get());
    }

    if (backendType.value() == executor::CacheTransceiverConfig::BackendType::UCX)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        mWrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
        TLLM_CHECK_WITH_INFO(
            mWrapperLibHandle != nullptr, "UCX wrapper library is not open correctly. error : %s", dlerror());
        auto load_sym = [](void* handle, char const* name)
        {
            void* ret = dllGetSym(handle, name);
            TLLM_CHECK_WITH_INFO(ret != nullptr,
                "Unable to load UCX wrapper library symbol, possible cause is that TensorRT LLM library is not "
                "built with UCX support, please rebuild in UCX-enabled environment.");
            return ret;
        };
        std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
        *(void**) (&makeUcxConnectionManager) = load_sym(mWrapperLibHandle, "makeUcxConnectionManager");
        mManager = makeUcxConnectionManager();
        TLLM_LOG_INFO("UCX Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::NIXL)
    {
        auto rnnState
            = mCacheState->hasRnnConfig() ? std::make_optional(mCacheState->getRnnCacheState()) : std::nullopt;
        mManager = std::make_unique<tensorrt_llm::executor::kv_cache::AgentConnectionManager>(
            mCacheTransBufferManagerPtrs, *mCacheState, "nixl", rnnState);
        TLLM_LOG_INFO("NIXL Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::MOONCAKE)
    {
        auto rnnState
            = mCacheState->hasRnnConfig() ? std::make_optional(mCacheState->getRnnCacheState()) : std::nullopt;
        mManager = std::make_unique<tensorrt_llm::executor::kv_cache::AgentConnectionManager>(
            mCacheTransBufferManagerPtrs, *mCacheState, "mooncake", rnnState);
        TLLM_LOG_INFO("MOONCAKE Connection Manager created");
    }
    else if (backendType.value() == executor::CacheTransceiverConfig::BackendType::MPI)
    {
        mMpiWorldComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mManager = std::make_unique<executor::kv_cache::MpiConnectionManager>(mMpiWorldComm);
        TLLM_LOG_INFO("MPI Connection Manager created");
    }
    else
    {
        TLLM_THROW("Unsupported cache transceiver backend type ");
    }

    auto makeFormatter = [cacheManager, isMLA, this]()
    {
        std::vector<kv_cache_manager::CacheTransBufferManager*> kvBufferPtrs;
        kvBufferPtrs.reserve(mCacheTransBufferManagers.size());
        for (auto& mgr : mCacheTransBufferManagers)
        {
            kvBufferPtrs.push_back(mgr.get());
        }
        return createCacheFormatter(cacheManager, kvBufferPtrs, isMLA);
    };

    auto makeRnnFormatter = [this]() -> std::unique_ptr<RnnCacheFormatter>
    {
        if (mRnnStateManager != nullptr && mRnnCacheTransBufferManager != nullptr)
        {
            return std::make_unique<RnnCacheFormatter>(mRnnStateManager, mRnnCacheTransBufferManager.get());
        }
        return nullptr;
    };

    auto makeCacheTransferLayer
        = [&]() { return CacheTransferLayer(*mCacheState, makeFormatter(), makeRnnFormatter()); };

    mCacheSender = std::make_unique<CacheSender>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer());
    mCacheReceiver = std::make_unique<CacheReceiver>(mManager.get(), worldConfig.getRank(), makeCacheTransferLayer());

    initializeCommState();
}

CacheTransceiver::~CacheTransceiver()
{
    if (mWrapperLibHandle)
    {
        std::lock_guard<std::mutex> lock(mDllMutex);
        dllClose(mWrapperLibHandle);
    }
}

void CacheTransceiver::initializeCommState()
{
    mCommState = std::addressof(mCacheSender->getCommState());
}

void CacheTransceiver::setContextState(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    auto contextState = std::make_unique<executor::DataTransceiverState>();
    contextState->setCommState(*mCommState);
    contextState->setCacheState(*mCacheState);
    if (!llmRequest->hasDraftTokens())
    {
        llmRequest->setContextPhaseParams(
            executor::ContextPhaseParams{{}, llmRequest->mRequestId, contextState.release(), std::nullopt});
    }
    else
    {
        llmRequest->setContextPhaseParams(executor::ContextPhaseParams{
            {}, llmRequest->mRequestId, contextState.release(), *llmRequest->getDraftTokens()});
    }
}

void CacheTransceiver::respondAndSendAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
    llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS);
    // If context phase params is already set, it means that the KV cache
    // transfer is already in progress.
    if (llmRequest->getContextPhaseParams().has_value())
    {
        if (llmRequest->getContextProgress() == nullptr)
        {
            TLLM_LOG_WARNING("Request %ld is already responding", llmRequest->mRequestId);
        }
        return;
    }
    setContextState(llmRequest);
    auto future = mCacheSender->sendAsync(*llmRequest);
    mSenderFutures.emplace_back(llmRequest, std::move(future));
}

void CacheTransceiver::respondAndSendLayerWise(
    RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
{
    for (auto const& llmRequest : requests)
    {
        TLLM_CHECK(llmRequest && llmRequest->isContextOnlyRequest());
        TLLM_CHECK(!llmRequest->getContextPhaseParams().has_value());
        llmRequest->setContextProgress(progress);
        TLLM_LOG_DEBUG("Request %ld is being sent layer-wise.", llmRequest->mRequestId);

        llmRequest->setState(LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS);
        setContextState(llmRequest.get());
        auto future = mCacheSender->sendAsync(*llmRequest);
        mSenderFutures.emplace_back(llmRequest.get(), std::move(future));
    }
}

void CacheTransceiver::requestAndReceiveSync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());
    {
        auto future = mCacheReceiver->receiveAsync(*llmRequest);
        future.get();
    }
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
}

void CacheTransceiver::requestAndReceiveAsync(LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest && llmRequest->isGenerationOnlyRequest());

    if (std::find_if(mRequesterFutures.begin(), mRequesterFutures.end(),
            [llmRequest](auto const& pair) { return pair.first->mRequestId == llmRequest->mRequestId; })
        != mRequesterFutures.end())
    {
        TLLM_LOG_WARNING("Request ID %zu is already in mRequestFutures.", llmRequest->mRequestId);
        return;
    }

    auto future = mCacheReceiver->receiveAsync(*llmRequest);
    mRequesterFutures.emplace_back(llmRequest, std::move(future));
    llmRequest->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS);
}

std::vector<LlmRequest::RequestIdType> gatherRequestIds(
    std::shared_ptr<CacheTransceiverComm> const& mComm, std::vector<LlmRequest::RequestIdType> const& requestIds)
{
    int localSize = static_cast<int>(requestIds.size());
    std::vector<int> sizes(mComm->getSize());
    std::vector<LlmRequest::RequestIdType> retData;
    if (useMPI())
    {
        mComm->allgather(&localSize, sizes.data(), 1, mpi::MpiType::kINT32);
        std::vector<int> displs(mComm->getSize());
        size_t totalSize = 0;
        for (int i = 0; i < mComm->getSize(); i++)
        {
            displs[i] = totalSize;
            totalSize += sizes[i];
        }
        retData.resize(totalSize);
        mComm->allgatherv(requestIds.data(), static_cast<int>(requestIds.size()), mpi::MpiType::kUINT64, retData.data(),
            sizes, displs, mpi::MpiType::kUINT64);
    }
    else
    {
        mComm->allgather(&localSize, std::ref(sizes), {});
        size_t totalSize = std::accumulate(sizes.begin(), sizes.end(), 0);
        retData.resize(totalSize);
        mComm->allgatherv(std::ref(requestIds), std::ref(retData), std::cref(sizes), {});
    }
    return retData;
}

void updateKVCacheTransferBW(std::shared_ptr<CacheTransceiverComm> const& mComm, LlmRequest* request)
{
    namespace su = executor::serialize_utils;
    int worldSize = mComm->getSize();

    std::ostringstream oStream;
    su::serialize(request->getKvCacheTransferStart(), oStream);
    su::serialize(request->getKvCacheTransferEnd(), oStream);

    auto str = oStream.str();
    std::vector<char> sendBuffer(str.begin(), str.end());
    auto sendBufferSize = sendBuffer.size();
    auto recvBufferSize = sendBufferSize * worldSize;
    std::vector<char> recvBuffer(recvBufferSize);

    if (useMPI())
    {
        mComm->allgather(sendBuffer.data(), recvBuffer.data(), sendBufferSize, mpi::MpiType::kCHAR);
    }
    else
    {
        mComm->allgather(std::ref(sendBuffer), std::ref(recvBuffer), {});
    }

    su::VectorWrapBuf<char> strbuf(recvBuffer);
    std::istream is(&strbuf);

    auto minStartTime = executor::RequestPerfMetrics::TimePoint::max();
    auto maxEndTime = executor::RequestPerfMetrics::TimePoint::min();

    for (int rank = 0; rank < worldSize; rank++)
    {
        minStartTime = std::min(su::deserialize<executor::RequestPerfMetrics::TimePoint>(is), minStartTime);
        maxEndTime = std::max(su::deserialize<executor::RequestPerfMetrics::TimePoint>(is), maxEndTime);
    }

    // Handle KV cache size separately - gather all sizes to the leader rank
    std::size_t localKVCacheSize = request->getKvCacheSize();
    std::vector<std::size_t> allKVCacheSizes(worldSize, 0);

    if (useMPI())
    {
        mComm->allgather(&localKVCacheSize, allKVCacheSizes.data(), 1, mpi::MpiType::kUINT64);
    }
    else
    {
        mComm->allgather(&localKVCacheSize, std::ref(allKVCacheSizes), {});
    }

    std::size_t totalKVCacheSize = 0;
    for (int rank = 0; rank < worldSize; rank++)
    {
        totalKVCacheSize += allKVCacheSizes[rank];
    }

    // Update the latest KV cache transfer time for leader rank
    if (mComm->getRank() == 0)
    {
        request->setKvCacheTransferStart(minStartTime);
        request->setKvCacheTransferEnd(maxEndTime);
        request->setKvCacheSize(totalKVCacheSize);
    }
}

RequestStatuses CacheTransceiver::checkContextTransferStatus(
    std::optional<int> const& atLeastRequestNum, bool markComplete)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::optional<int> senderFutureTimeoutMs = std::nullopt;
    // If blockAll is true, we want to block and not use a timeout
    if (!blockAll && mCacheTransceiverConfig.has_value())
    {
        senderFutureTimeoutMs = mCacheTransceiverConfig->getKvTransferSenderFutureTimeoutMs();
    }

    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupTPInDPComm : mGroupTensorParaComm;
    std::vector<LlmRequest::RequestIdType> contextCompleteRequestIds;
    for (auto&& [request, future] : mSenderFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            contextCompleteRequestIds.push_back(request->mRequestId);
        }
    }

    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;
    if ((syncComm) && syncComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(syncComm, contextCompleteRequestIds);
        for (auto&& requestId : gatherRequestIdVec)
        {
            frequencyMap[requestId]++;
        }
    }
    else
    {
        for (auto&& requestId : contextCompleteRequestIds)
        {
            frequencyMap[requestId]++;
        }
    }
    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());

    std::sort(freqVec.begin(), freqVec.end(),
        [](std::pair<LlmRequest::RequestIdType, int> const& left,
            std::pair<LlmRequest::RequestIdType, int> const& right) { return left.second > right.second; });
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((syncComm) ? syncComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
    }

    // Make sure there are at least atLeastRequestNum requests in toCompleteIdSet.
    // This will preserve the order of insertion for KVCache transfer requests.
    for (auto it = mSenderFutures.begin();
         atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()) && it != mSenderFutures.end(); ++it)
    {
        auto& [request, future] = *it;
        toCompleteIdSet.insert(request->mRequestId);
    }

    RequestStatuses requestsStatus{};

    // Complete all the requests in toCompleteIdSet
    for (auto it = mSenderFutures.begin(); it != mSenderFutures.end();)
    {
        auto& [request, future] = *it;
        if (blockAll || (toCompleteIdSet.find(request->mRequestId) != toCompleteIdSet.end()))
        {
            try
            {
                // Wait for up to a specified timeout
                auto status = future.wait_for(std::chrono::milliseconds(senderFutureTimeoutMs.value_or(0)));
                if (status == std::future_status::ready || !senderFutureTimeoutMs.has_value())
                {
                    future.get();
                    requestsStatus.completedRequestIds.insert(request->mRequestId);
                    if (markComplete)
                    {
                        request->setState(LlmRequestState::kDISAGG_CONTEXT_COMPLETE);
                    }
                    it = mSenderFutures.erase(it);
                }
                else if (status == std::future_status::timeout)
                {
                    TLLM_LOG_WARNING("Timed out waiting for context KV cache transfer after %d milliseconds.",
                        senderFutureTimeoutMs.value());
                    ++it;
                }
                else
                {
                    TLLM_LOG_ERROR(
                        "Future returned unexpected status for request %ld. Marking as error", request->mRequestId);

                    request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                    requestsStatus.errorRequestIds.insert(request->mRequestId);
                    it = mSenderFutures.erase(it);
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR(
                    "Error occurred during context transfer for request %ld: %s", request->mRequestId, e.what());
                request->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                requestsStatus.errorRequestIds.insert(request->mRequestId);
                it = mSenderFutures.erase(it);
            }
        }
        else
        {
            ++it;
        }
    }

    return requestsStatus;
}

void CacheTransceiver::checkGenTransferStatus(std::optional<int> const& atLeastRequestNum)
{
    bool blockAll = !atLeastRequestNum.has_value();
    std::vector<LlmRequest::RequestIdType> genTransferReadyRequestIds;
    for (auto&& [request, future] : mRequesterFutures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            genTransferReadyRequestIds.push_back(request->mRequestId);
        }
    }
    std::unordered_map<LlmRequest::RequestIdType, int> frequencyMap;

    std::vector<LlmRequest::RequestIdType> toBlockRequestIds;
    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupDataComm : mGroupComm;
    if ((syncComm) && syncComm->getSize() > 1)
    {
        auto gatherRequestIdVec = gatherRequestIds(syncComm, genTransferReadyRequestIds);
        for (auto&& requestId : gatherRequestIdVec)
        {
            frequencyMap[requestId]++;
        }
    }
    else
    {
        for (auto&& requestId : genTransferReadyRequestIds)
        {
            frequencyMap[requestId]++;
        }
    }

    std::vector<std::pair<LlmRequest::RequestIdType, int>> freqVec(frequencyMap.begin(), frequencyMap.end());

    std::sort(freqVec.begin(), freqVec.end(),
        [](std::pair<LlmRequest::RequestIdType, int> const& left,
            std::pair<LlmRequest::RequestIdType, int> const& right) { return left.second > right.second; });
    std::unordered_set<LlmRequest::RequestIdType> toCompleteIdSet;
    size_t idx = 0;
    while (atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= freqVec.size())
        {
            break;
        }
        toCompleteIdSet.insert(freqVec.at(idx).first);
        if (useMPI())
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                " checkGenTransferStatus at least from freqVec requestId: %zu ", freqVec.at(idx).first);
        }
        else
        {
            TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                " checkGenTransferStatus at least from freqVec requestId: %zu ", freqVec.at(idx).first);
        }
        idx++;
    }
    idx = 0;

    // insert order
    while (atLeastRequestNum.value_or(0) > static_cast<int>(toCompleteIdSet.size()))
    {
        if (idx >= mRequesterFutures.size())
        {
            break;
        }
        if (toCompleteIdSet.find(mRequesterFutures.at(idx).first->mRequestId) == toCompleteIdSet.end())
        {
            toCompleteIdSet.insert(mRequesterFutures.at(idx).first->mRequestId);
            if (useMPI())
            {
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    " checkGenTransferStatus at least from RequesterFuture requestId: %zu atLeastRequestNum:%d",
                    mRequesterFutures.at(idx).first->mRequestId, atLeastRequestNum.value_or(0));
            }
            else
            {
                TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                    " checkGenTransferStatus at least from RequesterFuture requestId: %zu atLeastRequestNum:%d",
                    mRequesterFutures.at(idx).first->mRequestId, atLeastRequestNum.value_or(0));
            }
        }
        idx++;
    }
    for (auto&& [requestId, freq] : freqVec)
    {
        if (freq == ((syncComm != nullptr) ? syncComm->getSize() : 1))
        {
            toCompleteIdSet.insert(requestId);
        }
        if (useMPI())
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ",
                requestId, freq);
        }
        else
        {
            TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                " checkGenTransferStatus freqVec requestId: %zu,freq:%d  ", requestId, freq);
        }
    }
    if (useMPI())
    {
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            " checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d ", toCompleteIdSet.size(),
            atLeastRequestNum.value_or(0));
    }
    else
    {
        TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
            " checkGenTransferStatus toCompleteIdSet size: %zu, atLeastRequestNum: %d ", toCompleteIdSet.size(),
            atLeastRequestNum.value_or(0));
    }
    for (auto it = mRequesterFutures.begin(); it != mRequesterFutures.end();)
    {
        if (blockAll || toCompleteIdSet.find(it->first->mRequestId) != toCompleteIdSet.end())
        {
            try
            {
                it->second.get();
                it->first->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);

                // Gather the kv cache transfer time from all workers and update to leader rank
                if (!common::getEnvKVCacheTimeOutputPath().empty())
                {
                    auto syncComm = mCacheState->getParallelConfig().mEnableAttentionDP ? mGroupDataComm : mGroupComm;
                    updateKVCacheTransferBW(syncComm, it->first);
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR(
                    "Error occurred during generation transfer for request %ld: %s", it->first->mRequestId, e.what());
                it->first->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            }
            if (useMPI())
            {
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***",
                    it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            }
            else
            {
                TLLM_LOG_DEBUG(tensorrt_llm::pg_utils::get_world_pg()->getRank(),
                    "**** it->first->mRequestId: %ld, context request ID: %ld ******** get feature ***",
                    it->first->mRequestId, it->first->getContextPhaseParams().value().getReqId());
            }
            it = mRequesterFutures.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool CacheTransceiver::checkGenTransferComplete() const
{
    return mRequesterFutures.empty();
}

bool CacheTransceiver::cancelRequest(LlmRequest* llmRequest)
{
    if (llmRequest->isContextOnlyRequest())
    {
        return mCacheSender->cancelRequest(*llmRequest);
    }
    else if (llmRequest->isGenerationOnlyRequest())
    {
        return mCacheReceiver->cancelRequest(*llmRequest);
    }
    return false;
}

} // namespace tensorrt_llm::batch_manager
""",
    'key-files/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp': r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/evictionPolicy.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/batch_manager/radixBlockTree.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <utility>

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tle = tensorrt_llm::executor;
using namespace tle::kv_cache;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager::eviction_policy;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

namespace
{

//! \brief Get all blocks in a sequence by traversing backwards from the last block.
//! \param lastBlock is a BlockPtr to the last block in the sequence to start traversal from
//! \return Vector of BlockPtr-s in sequence order
std::vector<BlockPtr> getAllSequenceBlocks(BlockPtr lastBlock)
{
    // First count the number of blocks to pre-allocate the vector
    auto currentBlock = lastBlock;
    size_t blockCount = 0;
    while (currentBlock != nullptr && currentBlock->getBlockId() != KVCacheBlock::kCachedBlocksRootId)
    {
        blockCount++;
        currentBlock = currentBlock->getPrevBlockInSeq();
    }

    if (blockCount == 0)
    {
        return {};
    }
    // Create and pre-allocate the vector with the correct size
    std::vector<BlockPtr> sequenceBlocks(blockCount);

    // Now traverse backwards and fill from the end
    currentBlock = lastBlock;
    size_t currentIndex = blockCount - 1;
    while (currentBlock != nullptr && currentBlock->getBlockId() != KVCacheBlock::kCachedBlocksRootId)
    {
        sequenceBlocks[currentIndex--] = currentBlock;
        currentBlock = currentBlock->getPrevBlockInSeq();
    }

    return sequenceBlocks;
}

// Compute maximum number of tokens that have been computed by prefill and generation.
// Accounts for chunked prefill to avoid storing state that hasn't been written to KV cache yet.
// We call LlmRequest::getContextRemainingLength to see how many tokens are still waiting to be computed in prefill.
// If this value is > 0 prefill is not finished yet, and number of computed tokens must be capped at the current context
// position. If it is == 0, we are in generation mode, and number of computed tokens equals number of unique tokens
// stored in request.
SizeType32 getMaterializedUniqueTokenCountForReuse(
    VecUniqueTokens const& uniqueTokens, tensorrt_llm::batch_manager::LlmRequest const& llmRequest)
{
    auto const totalUniqueTokenCount = static_cast<SizeType32>(uniqueTokens.size());
    if (llmRequest.getContextRemainingLength() > 0)
    {
        return std::min(totalUniqueTokenCount, llmRequest.getContextCurrentPosition());
    }
    return totalUniqueTokenCount;
}

// Compute number of tokens that can be stored for reuse. The last computed token is never stored in KV cache, hence
// cannot be stored for reuse. Number of tokens that can be stored for reuse is thus the greater of 0 or
// getMaterializedUniqueTokenCountForReuse() - 1.
SizeType32 getUsableUniqueTokenCountForReuse(
    VecUniqueTokens const& uniqueTokens, tensorrt_llm::batch_manager::LlmRequest const& llmRequest)
{
    auto const materializedUniqueTokenCount = getMaterializedUniqueTokenCountForReuse(uniqueTokens, llmRequest);
    return materializedUniqueTokenCount > 0 ? materializedUniqueTokenCount - 1 : 0;
}

} // namespace

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
KVCacheBlock::KVCacheBlock(IdType blockId, tk::KVCacheIndex blockIdx, SizeType32 windowSize)
    : mBlockId(blockId)
    , mMemoryPoolBlockIndex{blockIdx}
    , mRefCount(0)
    , mSchedulingRefCount(0)
    , mLookupNode{nullptr}
    , mWindowSize{std::numeric_limits<int>::max()}
    // sentinel: unattached; valid sizes are >= 1 or kRecurrentStates (-1)
    , mIsPlaceholder{false}
    , mFreeBlockIterator(std::nullopt)
    , mIsFull{false}
    , mPriority{executor::KvCacheRetentionConfig::kDefaultRetentionPriority}
    , mDurationMs{std::nullopt}
    , mExpirationTime{std::nullopt}
    , mHash{0}
{
}

BlockPtr KVCacheBlock::createPlaceholder(IdType blockId, SizeType32 windowSize)
{
    // Use an out-of-range pool index as sentinel; the mIsPlaceholder flag gates
    // getCacheBlockIndices to return nil so this index is never submitted to the GPU.
    auto block = std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex::nullIndex, windowSize);
    block->mIsPlaceholder = true;
    return block;
}

BlockPtr KVCacheBlock::createPlaceholder()
{
    // SWA on-demand placeholder form: never attached to the lookup tree, only occupies a
    // slot in the sequence's allocated-blocks list. The sentinel block ID ensures that any
    // accidental mAllBlocksById[id] lookup produces an obvious out-of-range failure.
    auto block = std::make_shared<KVCacheBlock>(kPlaceholderBlockId, tk::KVCacheIndex::nullIndex);
    block->mIsPlaceholder = true;
    return block;
}

bool KVCacheBlock::isPlaceholder() const
{
    return mIsPlaceholder;
}

void KVCacheBlock::startScheduling()
{
    mSchedulingRefCount = mRefCount;
}

KVCacheBlock::IdType KVCacheBlock::getBlockId() const
{
    return mBlockId;
}

NextBlockMap KVCacheBlock::getNextBlocks() const
{
    if (!mLookupNode)
    {
        return {};
    }
    NextBlockMap result;
    for (auto const& [key, block] : mLookupNode->getChildKeyValues(mWindowSize))
    {
        result.emplace(key, block);
    }
    return result;
}

void KVCacheBlock::attachToLookupNode(radix_block_tree::LookupNodePtr node, int windowSize)
{
    // Detach from any previous node first.
    if (mLookupNode)
    {
        auto const wasCleared = mLookupNode->clearValue(mWindowSize);
        TLLM_CHECK_WITH_INFO(wasCleared,
            "attachToLookupNode: block %d expected prior lookup slot to be occupied (clearValue returned false)",
            static_cast<int>(mBlockId));
    }
    // Assign fields AFTER trySetValue so local state is only updated on success.
    auto const wasInserted = node->trySetValue(windowSize, shared_from_this(), /*overwrite=*/false);
    TLLM_CHECK_WITH_INFO(wasInserted,
        "attachToLookupNode: block %d found lookup slot already occupied by another block", static_cast<int>(mBlockId));
    mLookupNode = std::move(node);
    mWindowSize = windowSize;
}

void KVCacheBlock::detachFromLookupNode()
{
    if (!mLookupNode)
    {
        return;
    }
    // clearValue triggers the cascade-prune up through empty ancestor nodes automatically.
    auto const wasCleared = mLookupNode->clearValue(mWindowSize);
    TLLM_CHECK_WITH_INFO(wasCleared,
        "detachFromLookupNode: block %d expected lookup slot to be occupied (clearValue returned false)",
        static_cast<int>(mBlockId));
    mLookupNode = nullptr;
    mWindowSize = std::numeric_limits<int>::max();
}

void KVCacheBlock::setAsRoot(radix_block_tree::LookupNodePtr rootNode, int windowSize)
{
    mLookupNode = rootNode;
    mWindowSize = windowSize;
    // Store the root block itself in the root node so that direct children can find it
    // via getPrevBlock() (root->getParentNode() returns nullptr, so the chain stops here).
    auto const wasUpdated = rootNode->trySetValue(windowSize, shared_from_this(), /*overwrite=*/true);
    TLLM_LOG_DEBUG("setAsRoot: block %d wired to root slot for windowSize=%d (wasUpdated=%d)",
        static_cast<int>(mBlockId), windowSize, static_cast<int>(wasUpdated));
}

// This is a logical index. In memory pool, 1 in mMemoryPoolBlockIndex is strided by num_layers * kv_num. (see
// WindowBlockManager::setOffsets)
tk::KVCacheIndex::UnderlyingType KVCacheBlock::getMemoryPoolBlockIndex() const
{
    return mMemoryPoolBlockIndex.get();
}

std::vector<MmKey> KVCacheBlock::getExtraKeys() const
{
    return mBlockKey.extraKeys;
}

bool KVCacheBlock::isPrimary() const
{
    TLLM_CHECK_WITH_INFO(!isPlaceholder(), "Not expected to call isPrimary() on placeholder block");
    return mMemoryPoolBlockIndex.isPrimary();
}

void KVCacheBlock::swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock)
{
    TLLM_CHECK(!isPlaceholder() && !otherBlock->isPlaceholder());
    std::swap(mMemoryPoolBlockIndex, otherBlock->mMemoryPoolBlockIndex);
}

void KVCacheBlock::incRefCount()
{
    mRefCount++;
}

void KVCacheBlock::decRefCount()
{
    TLLM_CHECK_WITH_INFO(
        hasRefs(), "Can't remove link from block (id=%d) that is not allocated", static_cast<int>(mBlockId));
    mRefCount--;
}

void KVCacheBlock::decSchedulingRefCount()
{
    TLLM_CHECK_WITH_INFO(hasSchedulingRefs(), "Can't remove link from block that is not allocated");
    mSchedulingRefCount--;
}

bool KVCacheBlock::hasRefs() const
{
    return mRefCount > 0;
}

bool KVCacheBlock::isShared() const
{
    // Block is considered shared if it has multiple references or is registered in the
    // lookup tree (i.e., it is cached for reuse by future requests).
    // Note: mCachedBlocksRoot also has mLookupNode set (via setAsRoot), but it is never
    // placed in the eviction queue — enforced by an assertion in LRUEvictionPolicy::releaseBlock.
    return mRefCount > 1 || mLookupNode != nullptr;
}

bool KVCacheBlock::hasSchedulingRefs() const
{
    return mSchedulingRefCount > 0;
}

void KVCacheBlock::setBlockKey(BlockKey const& blockKey, bool isFull)
{
    mBlockKey = blockKey;
    mIsFull = isFull;
}

BlockKey KVCacheBlock::getBlockKey()
{
    return mBlockKey;
}

void KVCacheBlock::setPriority(executor::RetentionPriority priority)
{
    mPriority = priority;
}

executor::RetentionPriority KVCacheBlock::getPriority() const
{
    return mPriority;
}

std::optional<std::chrono::milliseconds> KVCacheBlock::getDurationMs() const
{
    return mDurationMs;
}

void KVCacheBlock::setDurationMs(std::optional<std::chrono::milliseconds> durationMs)
{
    mDurationMs = durationMs;
}

void KVCacheBlock::setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime)
{
    mExpirationTime = expirationTime;
}

std::optional<std::chrono::steady_clock::time_point::duration> KVCacheBlock::getExpirationTime() const
{
    return mExpirationTime;
}

void KVCacheBlock::setHash(size_t hash)
{
    mHash = hash;
}

void KVCacheBlock::setHash()
{
    mHash = BlockKeyHasher()(mBlockKey, mPrevBlockInSeq ? mPrevBlockInSeq->getHash() : 0);
}

size_t KVCacheBlock::getHash() const
{
    return mHash;
}

VecUniqueTokens const& KVCacheBlock::getUniqueTokens() const
{
    return mBlockKey.uniqueTokens;
}

BlockPtr KVCacheBlock::getPrevBlock() const
{
    if (!mLookupNode)
    {
        return nullptr;
    }
    auto parentNode = mLookupNode->getParentNode();
    if (!parentNode)
    {
        // This block is the root (no parent node), so it has no parent block.
        return nullptr;
    }
    auto optBlock = parentNode->getValue(mWindowSize);
    return optBlock.value_or(nullptr);
}

BlockPtr const& KVCacheBlock::getPrevBlockInSeq() const
{
    return mPrevBlockInSeq;
}

void KVCacheBlock::setPrevBlockInSeq(BlockPtr prevBlock)
{
    mPrevBlockInSeq = std::move(prevBlock);
}

void KVCacheBlock::addNextBlock(BlockKey const& blockKey, BlockPtr block)
{
    if (!mLookupNode)
    {
        return;
    }
    // Find existing child node or create a new one, then wire the block into it.
    auto childNode = mLookupNode->findOrInsertChild(blockKey, mLookupNode);
    // Only attach if there is no block already stored for this window size (matches old
    // behaviour: addNextBlock was a no-op when the key already existed in mNextBlocks).
    auto existing = childNode->getValue(mWindowSize);
    if (!existing.has_value())
    {
        block->attachToLookupNode(childNode, mWindowSize);
    }
}

std::tuple<bool, SizeType32, BlockPtr> KVCacheBlock::findMatchingBlock(
    BlockKey const& blockKey, bool enablePartialReuse, bool copyOnPartialReuse) const
{
    if (!mLookupNode || blockKey.uniqueTokens.empty())
    {
        return {false, 0, nullptr};
    }

    // Exact match
    auto exactMatch = mLookupNode->findMatchingNode(blockKey);
    if (exactMatch.has_value())
    {
        auto optBlock = exactMatch->node->getValue(mWindowSize);
        if (optBlock.has_value() && *optBlock)
        {
            auto block = *optBlock;
            return {!block->isFull(), static_cast<SizeType32>(blockKey.uniqueTokens.size()), block};
        }
        return {false, 0, nullptr};
    }

    // Partial match (sorted longest-first by findPartiallyMatchingNodes)
    if (enablePartialReuse)
    {
        auto partialMatches = mLookupNode->findPartiallyMatchingNodes(blockKey);
        for (auto const& match : partialMatches)
        {
            auto optBlock = match.node->getValue(mWindowSize);
            if (!optBlock.has_value() || !(*optBlock))
            {
                continue;
            }
            auto block = *optBlock;
            if (copyOnPartialReuse || (!block->hasRefs() && block->isLeaf()))
            {
                return {true, static_cast<SizeType32>(match.key.uniqueTokens.size()), block};
            }
        }
    }

    return {false, 0, nullptr};
}

void KVCacheBlock::freeLeafBlock()
{
    // assure that this is a leaf block
    TLLM_CHECK(isLeaf());
    // Detach from the lookup tree; cascade pruning removes empty ancestor nodes.
    detachFromLookupNode();
}

void KVCacheBlock::removeNextBlock(BlockKey const& blockKey)
{
    if (mLookupNode)
    {
        // clearNode removes the child entry and fires cascade pruning upward if the child
        // node becomes empty after the removal.
        auto const wasCleared = mLookupNode->clearNode(blockKey);
        if (!wasCleared)
        {
            TLLM_LOG_DEBUG("removeNextBlock: key not found for block %d; node may have been pruned already",
                static_cast<int>(mBlockId));
        }
    }
}

// Iterative DFS over the subtree rooted at this block's children.
//
// Algorithm:
//   1. Push immediate children onto a stack and do DFS, collecting every
//      reachable descendant in pre-order (parent before children).
//   2. Detach in *reverse* order (children before parents).  This is
//      required because detachFromLookupNode() triggers cascade pruning:
//      when a node becomes empty (no value, no children) it is removed from
//      its parent.  If we detached in collection order (parents first), a
//      parent node could be cascade-pruned away before we had a chance to
//      look up its children in step 1.  By detaching leaves first, cascade
//      propagation only moves upward after all descendants are already gone.
void KVCacheBlock::detachDescendantsFromLookupTree()
{
    if (!mLookupNode)
    {
        return;
    }
    std::vector<BlockPtr> descendants;
    std::vector<BlockPtr> stack;
    for (auto const& [key, block] : mLookupNode->getChildKeyValues(mWindowSize))
    {
        stack.push_back(block);
    }
    while (!stack.empty())
    {
        auto current = std::move(stack.back());
        stack.pop_back();
        if (current->mLookupNode)
        {
            for (auto const& [key, block] : current->mLookupNode->getChildKeyValues(current->mWindowSize))
            {
                stack.push_back(block);
            }
        }
        TLLM_LOG_DEBUG("KVCacheBlock::detachDescendantsFromLookupTree - detaching block %d", current->getBlockId());
        descendants.push_back(std::move(current));
    }
    // Detach leaves first so cascade-prune works correctly.
    for (auto it = descendants.rbegin(); it != descendants.rend(); ++it)
    {
        (*it)->detachFromLookupNode();
    }
}

void KVCacheBlock::freeBlockAndAllDescendants()
{
    detachDescendantsFromLookupTree();
    detachFromLookupNode();
}

bool KVCacheBlock::isFull() const
{
    return mIsFull;
}

bool KVCacheBlock::isLeaf() const
{
    return !mLookupNode || !mLookupNode->hasChildren();
}

// This function calculates the number of block a layer should have, given
// the total free memory and the window size of each layer.
// For example, if we have 1 layer of window size 1024, and 2 layer of window
// size 2048, and 3 layers of 4096.
// Each layer of window size 1024 should have
//     1024 / (1024 + 2048 * 2 + 4096 * 3) proportion of the total blocks.
// Each layer of window size 2048 should have
//     2048 / (1024 + 2048 * 2 + 4096 * 3) proportion of the total blocks.
// Each layer of window size 4096 should have
//     4096 / (1024 + 2048 * 2 + 4096 * 3) proportion of the total blocks.
// NOTE: Currently the use of this function is not used for
// BaseKVCacheManager::calculateMaxNumBlocks because the we want to first
// achieve identical performance as assuming all layers as full attention.
std::map<SizeType32, float> BlockManager::calculateWindowSizeToShare(
    std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers,
    std::map<SizeType32, SizeType32> const& windowSizeToCacheSizePerToken)
{
    if (windowSizeToLayers.size() == 1)
    {
        return {{windowSizeToLayers.begin()->first, 1.0f}};
    }

    std::map<SizeType32, float> windowSizeToContribution;

    SizeType32 cacheSizePerTokenTotal
        = std::accumulate(windowSizeToCacheSizePerToken.begin(), windowSizeToCacheSizePerToken.end(), SizeType32{0},
            [](auto sum, auto const& windowSize) { return sum + windowSize.second; });
    for (auto const& [windowSize, cacheSizePerToken] : windowSizeToCacheSizePerToken)
    {
        auto const cacheSizeWeight = static_cast<float>(cacheSizePerToken) / cacheSizePerTokenTotal;
        windowSizeToContribution[windowSize] = cacheSizeWeight;
    }

    for (auto const& [windowSize, _] : windowSizeToLayers)
    {
        windowSizeToContribution.at(windowSize) *= windowSize;
    }
    auto const windowSizesTotalSum = std::accumulate(windowSizeToContribution.begin(), windowSizeToContribution.end(),
        0.0, [](auto sum, auto const& windowSize) { return sum + windowSize.second; });

    std::map<SizeType32, float> windowSizeToShare;
    for (auto const& [windowSize, windowSizeSum] : windowSizeToContribution)
    {
        float const fraction = windowSizeSum / windowSizesTotalSum;
        TLLM_CHECK(0.0f < fraction && fraction <= 1.0f);
        windowSizeToShare[windowSize] = fraction;
    }
    auto total = std::accumulate(windowSizeToShare.begin(), windowSizeToShare.end(), 0.0f,
        [](auto sum, auto const& windowSize) { return sum + windowSize.second; });
    TLLM_CHECK(total == 1.0f);
    return windowSizeToShare;
}

BlockManager::BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, CudaStreamPtr stream,
    SizeType32 maxSequenceLength, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    nvinfer1::DataType dtype, SizeType32 sinkBubbleLength, SizeType32 chunkSize, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
    std::optional<BaseAgentConfig> agentConfig, bool enableIndexerKCache, SizeType32 indexerKCacheQuantBlockSize,
    SizeType32 indexerKCacheIndexHeadDim, bool indexerKCacheUseFp4,
    std::optional<LinearAttentionMetadata> linearAttentionMetadata)
    : mNumLayers{static_cast<SizeType32>(numKvHeadsPerLayer.size())}
    , mTokensPerBlock{tokensPerBlock}
    , mEventManager{std::move(eventManager)}
    , mStream{stream}
    , mCacheType{cacheType}
    , mIsEnableIndexerKCache{enableIndexerKCache}
    , mIndexerKCacheQuantBlockSize{indexerKCacheQuantBlockSize}
    , mIndexerKCacheIndexHeadDim{indexerKCacheIndexHeadDim}
    , mIndexerKCacheUseFp4{indexerKCacheUseFp4}
    , mLinearAttentionMetadata{linearAttentionMetadata}
{
    if (mLinearAttentionMetadata.has_value())
    {
        TLLM_CHECK_WITH_INFO(enablePartialReuse == false, "Partial reuse is not supported with linear attention");
        if (mLinearAttentionMetadata->hasRecurrentStatesCache())
        {
            TLLM_CHECK(mLinearAttentionMetadata->statesSnapshotInterval % mTokensPerBlock == 0);
            // Enforce that a full-attention window (windowSize == maxSequenceLength) must be present
            // alongside kRecurrentStates.
            TLLM_CHECK_WITH_INFO(blocksPerWindow.count(maxSequenceLength) > 0,
                "kRecurrentStates window size requires a full-attention window size (== maxSequenceLength=%d) "
                "to be present alongside it.",
                maxSequenceLength);
        }
    }
    if (agentConfig.has_value())
        mLoopbackAgent = makeLoopbackAgent("nixl", &agentConfig.value());
    else
        mLoopbackAgent = nullptr;

    auto const uniqueWindowSizeToLayers
        = BaseKVCacheManager::groupLayersByWindowSize(maxAttentionWindowVec, mNumLayers);

    TLLM_CHECK_WITH_INFO(kvCacheConnectorManager == nullptr || uniqueWindowSizeToLayers.size() == 1,
        "KV Cache Connector is not supported with multiple window sizes");

    auto const numUniqueWindowSizes = static_cast<SizeType32>(uniqueWindowSizeToLayers.size());

    mIsVariableWindow = numUniqueWindowSizes > 1;
    mIsVariableGQA = std::unordered_set(numKvHeadsPerLayer.begin(), numKvHeadsPerLayer.end()).size() > 1;

    mLayerToWindowSize.resize(mNumLayers);
    for (auto const& [windowSize, layersWithWindowSize] : uniqueWindowSizeToLayers)
    {
        if (windowSize > maxSequenceLength)
        {
            TLLM_LOG_WARNING("[kv cache manager] window size %d is greater than max sequence length %d", windowSize,
                maxSequenceLength);
        }
        for (auto& layerIdx : layersWithWindowSize)
        {
            mLayerToWindowSize.at(layerIdx) = windowSize;
        }
        auto const [allottedPrimaryBlocks, allottedSecondaryBlocks] = blocksPerWindow.at(windowSize);
        TLLM_CHECK(allottedPrimaryBlocks > 0); // You can't have a model with negative primary blocks...

        // Compute numPlaceholderBlocks for kRecurrentStates managers: the difference between the
        // full-attention manager's primary block count and this manager's primary block count.
        SizeType32 numPlaceholderBlocks = 0;
        if (LinearAttentionMetadata::hasRecurrentStatesCache(windowSize))
        {
            if (linearAttentionMetadata.has_value() && linearAttentionMetadata->numPlaceholderBlocks.has_value())
            {
                numPlaceholderBlocks = *linearAttentionMetadata->numPlaceholderBlocks;
                TLLM_CHECK_WITH_INFO(numPlaceholderBlocks >= 0,
                    "LinearAttentionMetadata::numPlaceholderBlocks must be >= 0, got %d", numPlaceholderBlocks);
            }
            else
            {
                auto const [fullPrimaryBlocks, unusedSecondaryBlocks] = blocksPerWindow.at(maxSequenceLength);
                numPlaceholderBlocks
                    = fullPrimaryBlocks; // The full attention runs out of blocks before we use all placeholders.
            }
        }

        mWindowBlockManagers.try_emplace(SizeType32(windowSize), dtype, windowSize, layersWithWindowSize,
            numKvHeadsPerLayer, sizePerHead, tokensPerBlock,
            /*isSWA=*/(windowSize < maxSequenceLength) && (windowSize >= 0), allottedPrimaryBlocks,
            allottedSecondaryBlocks, maxNumSequences, stream, cacheType, secondaryOffloadMinPriority, mEventManager,
            enablePartialReuse, copyOnPartialReuse, kvCacheConnectorManager, mLookupTree, mLoopbackAgent,
            enableIndexerKCache, indexerKCacheQuantBlockSize, indexerKCacheIndexHeadDim, indexerKCacheUseFp4,
            LinearAttentionMetadata::hasLinearCache(windowSize) ? linearAttentionMetadata : std::nullopt,
            numPlaceholderBlocks);
    }

    auto const numAllPools = getNumPools();
    mAbsolutePoolToWindowSize.reserve(numAllPools);
    mAbsolutePoolToRelativePoolIndex.reserve(numAllPools);
    auto absolutePoolsOffset = SizeType32{0};
    for (auto const& [windowSize, manager] : mWindowBlockManagers)
    {
        auto const numPools = manager.getNumPools();
        for (auto i = 0; i < numPools; ++i)
        {
            mAbsolutePoolToWindowSize.push_back(windowSize);
            mAbsolutePoolToRelativePoolIndex.push_back(i);
        }
        // SWA allocates blocks linearly, and we need as many blocks as full attention,
        // where full attention has windowSize = maxSequenceLength.
        auto const maxTokenNum = std::max(windowSize, maxSequenceLength) + sinkBubbleLength;
        // For sliding window attention, we at most hold chunk_size + window_size number of tokens.
        // For full attention, all tokens are held.
        auto const maxTokensHeld
            = windowSize < maxSequenceLength ? std::min(maxSequenceLength, maxTokenNum + chunkSize) : maxTokenNum;
        auto const maxBlocksPerSeq = tc::ceilDiv(maxTokensHeld, tokensPerBlock);
        auto const [allottedPrimaryBlocks, allottedSecondaryBlocks] = blocksPerWindow.at(windowSize);
        mWindowSizeToMetadata[windowSize]
            = WindowSizeMetadata{allottedPrimaryBlocks, allottedSecondaryBlocks, absolutePoolsOffset, numPools,
                maxTokenNum, maxBlocksPerSeq, manager.getMaxNumBlocks(), windowSize, manager.isSWA()};
        TLLM_LOG_INFO(
            "Max KV cache blocks per sequence: %d [window size=%d], tokens per block=%d, primary blocks=%d, secondary "
            "blocks=%d, max sequence length=%d",
            maxBlocksPerSeq, windowSize, tokensPerBlock, allottedPrimaryBlocks, allottedSecondaryBlocks,
            maxSequenceLength);
        TLLM_LOG_DEBUG(
            "%s Metadata: %s", manager.getLogPrefix().c_str(), mWindowSizeToMetadata[windowSize].toString().c_str());
        absolutePoolsOffset += numPools;
    }

    TLLM_CHECK_WITH_INFO(mWindowBlockManagers.size() == mWindowSizeToMetadata.size()
            && std::equal(mWindowBlockManagers.cbegin(), mWindowBlockManagers.cend(), mWindowSizeToMetadata.cbegin(),
                mWindowSizeToMetadata.cend(),
                [](auto const& window1, auto const& window2) { return window1.first == window2.first; }),
        "Iteration order of window sizes between mWindowBlockManagers and mWindowSizeToMetadata *must* be ensured. "
        "Maybe you tried changing either of them to an std::unordered_map?");
}

WindowBlockManager::WindowBlockManager(nvinfer1::DataType dtype, SizeType32 windowSize,
    std::vector<SizeType32> const& managedLayers, std::vector<SizeType32> const& numKvHeadsPerLayer,
    SizeType32 sizePerHead, SizeType32 tokensPerBlock, bool isSWA, SizeType32 blocksInPrimaryPool,
    SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream,
    CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
    radix_block_tree::UnifiedBlockTree& lookupTree, std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent,
    bool enableIndexerKCache, SizeType32 indexerKCacheQuantBlockSize, SizeType32 indexerKCacheIndexHeadDim,
    bool indexerKCacheUseFp4, std::optional<LinearAttentionMetadata> linearAttentionMetadata,
    SizeType32 numPlaceholderBlocks)
    : mDataType{dtype}
    , mWindowSize{windowSize}
    , mNumPrimaryBlocks{blocksInPrimaryPool}
    , mNumSecondaryBlocks{blocksInSecondaryPool}
    , mBufferManager{std::move(stream)}
    , mSchedulingNumFreeBlocks{0}
    , mTokensPerBlock{tokensPerBlock}
    , mIsSWA{isSWA}
    , mLookupTree{&lookupTree}
    // Use an out-of-range pool index for the dummy root block; it is never submitted to the GPU.
    // The illegal value (INT32_MAX) ensures accidental use triggers an obvious OOB failure.
    , mCachedBlocksRoot{std::make_shared<KVCacheBlock>(KVCacheBlock::kCachedBlocksRootId,
          tk::KVCacheIndex{std::numeric_limits<tk::KVCacheIndex::UnderlyingType>::max()})}
    , mCacheType{cacheType}
    , mEventManager(std::move(eventManager))
    , mLoopbackAgent{loopbackAgent}
    , mTransferManager{std::make_shared<KVCacheTransferManager>(mBufferManager, mLoopbackAgent)}
    , mAllocTotalBlocks{0}
    , mAllocNewBlocks{0}
    , mFullReusedBlocks{0}
    , mPartialReusedBlocks{0}
    , mReusedBlocks{0}
    , mReusedUniqueBlocks{0}
    , mMissedBlocks{0}
    , mGenAllocBlocks{0}
    , mKVFactor{(mCacheType == CacheType::kSELFKONLY
                    || (linearAttentionMetadata.has_value() && linearAttentionMetadata->hasRecurrentStatesCache()))
              ? 1
              : 2}
    , mLogPrefix{tensorrt_llm::common::fmtstr("BlockManager[windowSize=%d]", mWindowSize)}
    , mReusedTokens{0.0}
    , mTotalInputTokens{0.0}
    , mEnablePartialReuse{enablePartialReuse}
    , mCopyOnPartialReuse{copyOnPartialReuse}
    , mKvCacheConnectorManager{std::move(kvCacheConnectorManager)}
    , mEnableIndexerKCache{enableIndexerKCache}
    , mIndexerKCacheQuantBlockSize{indexerKCacheQuantBlockSize}
    , mIndexerKCacheIndexHeadDim{indexerKCacheIndexHeadDim}
    , mIndexerKCacheUseFp4{indexerKCacheUseFp4}
    , mLinearAttentionMetadata{std::move(linearAttentionMetadata)}
{
    TLLM_LOG_DEBUG("Creating WindowBlockManager for windowSize=%d", windowSize);
    std::map<SizeType32, SizeType32> numLayersPerPool;

    for (auto const layerIdx : managedLayers)
    {
        auto const& layerIndexWithinPool = numLayersPerPool[numKvHeadsPerLayer.at(layerIdx)]++;
        mLayerToIndexWithinPool[layerIdx] = layerIndexWithinPool;
    }

    auto numEltsPerContainer = getNumEltsPerContainer();
#ifdef ENABLE_FP4
    if (numEltsPerContainer == 2)
    {
        TLLM_CHECK_WITH_INFO(sizePerHead % 2 == 0, "sizePerHead must be divisible by 2 for 4-bit KV cache.");
    }
#endif

    size_t poolIndex = 0;
    for (auto const [numKvHeads, numLayers] : numLayersPerPool)
    {
        for (auto const layerIdx : managedLayers)
        {
            if (numKvHeadsPerLayer.at(layerIdx) == numKvHeads)
            {
                mLayerToPoolIndex[layerIdx] = poolIndex;
            }
        }
        if (isRecurrentState())
        {
            TLLM_CHECK(numLayersPerPool.size() == 1);
            auto bytesPerElement = common::getDTypeSize(mDataType);
            KVCacheBlockPool pool(numLayers, /*kvFactor=*/1, /*numKvHeads=*/-1,
                /*sizePerHead=*/-1, tokensPerBlock);
            pool.blockSize = mLinearAttentionMetadata->allRecurrentStatesBytes / bytesPerElement;
            mPools.push_back(std::move(pool));
        }
        else
        {
            mPools.emplace_back(numLayers, mKVFactor, numKvHeads, sizePerHead / numEltsPerContainer, tokensPerBlock);
        }
        ++poolIndex;
    }

#ifdef ENABLE_FP4
    // TODO(miovine): make the block size configurable. Should we have an additional argument
    // to specify FP4 related parameters (scale dtypes, etc)? This can also be passed
    // in the constructor.
    constexpr SizeType32 kQuantBlockSizeNVFP4 = 16;
    if (dtype == nvinfer1::DataType::kFP4)
    {
        createBlockScalePools(kQuantBlockSizeNVFP4);
    }
#endif

    if (mEnableIndexerKCache)
    {
        createIndexerKCachePools();
    }

    // Create free blocks
    mAllBlocksById.reserve(blocksInPrimaryPool + blocksInSecondaryPool);
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInPrimaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex{blockId, false}));
    }
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInSecondaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(
            std::make_shared<KVCacheBlock>(blocksInPrimaryPool + blockId, tk::KVCacheIndex{blockId, true}));
    }
    mAllocatedBlocksPerSeq.reserve(maxNumSequences);

    // Pre-allocate placeholder blocks when this is a recurrent-state (linear attention) manager paired with
    // a full-attention manager. Placeholder IDs start at -2 (since -1 is reserved for kCachedBlocksRootId).
    // mAllPlaceholderBlocksById is indexed by abs(blockId): index 0 and 1 are unused (nullptr),
    // index abs(blockId) holds the block with that negative blockId.
    if (numPlaceholderBlocks > 0)
    {
        TLLM_LOG_DEBUG(
            "%s::ctor - pre-allocating %d placeholder blocks with IDs in range [%d, %d] for recurrent-state manager",
            mLogPrefix.c_str(), numPlaceholderBlocks, KVCacheBlock::kCachedBlocksRootId - 1 - numPlaceholderBlocks,
            KVCacheBlock::kCachedBlocksRootId - 2);
        TLLM_CHECK_WITH_INFO(isRecurrentState(),
            "numPlaceholderBlocks > 0 is only supported for recurrent-state (kRecurrentStates) managers");
        mAllPlaceholderBlocksById.resize(numPlaceholderBlocks + 2, nullptr);
        for (SizeType32 i = 0; i < numPlaceholderBlocks; ++i)
        {
            KVCacheBlock::IdType const placeholderBlockId
                = KVCacheBlock::kCachedBlocksRootId - 1 - static_cast<KVCacheBlock::IdType>(i);
            auto block = KVCacheBlock::createPlaceholder(placeholderBlockId, windowSize);
            mAllPlaceholderBlocksById[static_cast<size_t>(-placeholderBlockId)] = block;
        }

        auto policy = std::make_shared<LRUEvictionPolicy>();
        policy->initialize(mAllBlocksById, {blocksInPrimaryPool, blocksInSecondaryPool}, secondaryOffloadMinPriority);
        policy->initializePlaceholders(mAllPlaceholderBlocksById);
        mEvictionPolicy = policy;
    }
    else
    {
        mEvictionPolicy = std::make_shared<LRUEvictionPolicy>();
        mEvictionPolicy->initialize(
            mAllBlocksById, {blocksInPrimaryPool, blocksInSecondaryPool}, secondaryOffloadMinPriority);
    }
    if (mEventManager)
    {
        mEventManager->enqueueCreatedEvent({blocksInPrimaryPool, blocksInSecondaryPool}, mWindowSize);
    }

    // Wire the dummy root block into the shared lookup tree so that direct children
    // can navigate to it via getPrevBlock() and blockInRadixTree() returns true for them.
    mCachedBlocksRoot->setAsRoot(mLookupTree->getRoot(), mWindowSize);
}

WindowBlockManager::~WindowBlockManager()
{
    double reusedUniqueBlocksPercentage = mAllocNewBlocks == 0
        ? 0.0
        : static_cast<double>(mReusedUniqueBlocks) / static_cast<double>(mAllocNewBlocks) * 100.0;
    double cacheHitRate = (mReusedBlocks + mMissedBlocks) == 0 ? 0.0
                                                               : static_cast<double>(mReusedBlocks)
            / (static_cast<double>(mReusedBlocks) + static_cast<double>(mMissedBlocks));
    TLLM_LOG_DEBUG("%s - total allocated blocks:              %d  ", mLogPrefix.c_str(), mAllocTotalBlocks);
    TLLM_LOG_DEBUG("%s - allocated new blocks:                %d  ", mLogPrefix.c_str(), mAllocNewBlocks);
    TLLM_LOG_DEBUG("%s - missed blocks:                       %d  ", mLogPrefix.c_str(), mMissedBlocks);
    TLLM_LOG_DEBUG("%s - reused blocks:                       %d  ", mLogPrefix.c_str(), mReusedBlocks);
    TLLM_LOG_DEBUG("%s - reused unique blocks:                %d  ", mLogPrefix.c_str(), mReusedUniqueBlocks);
    TLLM_LOG_DEBUG(
        "%s - reused unique blocks percentage (%%): %.2f ", mLogPrefix.c_str(), reusedUniqueBlocksPercentage);
    TLLM_LOG_DEBUG("%s - cache hit rate:                      %.2f ", mLogPrefix.c_str(), cacheHitRate);
    TLLM_LOG_DEBUG("%s - reused tokens:                       %.0f ", mLogPrefix.c_str(), mReusedTokens);
    TLLM_LOG_DEBUG("%s - reused tokens percentage (%%):        %.2f ", mLogPrefix.c_str(),
        mTotalInputTokens == 0.0 ? 0.0 : 100.0 * mReusedTokens / mTotalInputTokens);
}

bool BlockManager::verifyQueueIntegrity(SizeType32 windowSize) const
{
    return mWindowBlockManagers.at(windowSize).verifyQueueIntegrity();
}

bool WindowBlockManager::verifyQueueIntegrity() const
{
    return mEvictionPolicy->verifyQueueIntegrity();
}

[[nodiscard]] BlockPtr WindowBlockManager::getBlockById(KVCacheBlock::IdType blockId) const
{
    if (blockId >= 0)
    {
        return mAllBlocksById.at(blockId);
    }
    // Negative blockIds are placeholder blocks. mAllPlaceholderBlocksById is indexed by abs(blockId).
    auto const idx = static_cast<size_t>(-blockId);
    if (idx >= mAllPlaceholderBlocksById.size() || blockId == KVCacheBlock::kCachedBlocksRootId)
    {
        return nullptr;
    }
    return mAllPlaceholderBlocksById[idx];
}

void BlockManager::storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest)
{
    // Iterate in descending window-size order (largest/full-attention windows first).
    // This guarantees that the Stored event for the full-attention window is committed
    // before flushRemovedEvents fires for SWA windows, preserving the per-window
    // ordering guarantee: Removed events precede the Stored event for the same window,
    // and Stored(full) is not interleaved with Removed(SWA).
    for (auto it = mWindowBlockManagers.rbegin(); it != mWindowBlockManagers.rend(); ++it)
    {
        it->second.storeContextBlocks(sequence, llmRequest);
    }
}

void WindowBlockManager::storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest)
{
    // Store fully-filled context blocks (both SWA and non-SWA) in the reuse trie so
    // that OOW blocks are already in the trie before detachFrontBlock replaces them
    // with placeholders. storeBlocks advances past placeholder slots without re-
    // inserting, and continues past evicted SWA anchors so later in-window blocks
    // remain reusable through value-less trie nodes.
    //
    // Unlike getUsableUniqueTokenCountForReuse (which caps at contextCurrentPosition when
    // prefill is not complete), here we key by uniqueTokens.size() - 1.  The last token
    // is not yet materialized, but all full blocks before it have been written to KV
    // cache during the context phase, so they are safe to store for reuse.  Callers may
    // invoke storeContextBlocks immediately after addSequenceBatch (before
    // simulatePrefillCompletion is used in tests, or before the first generation step in
    // production) and rely on the full context being stored.
    int constexpr beamIdx = 0; // no need to consider more than one beam for input tokens
    auto cacheBlockIds = sequence.getCacheBlockIds(mWindowSize);
    auto const& uniqueTokens = llmRequest.getUniqueTokens(beamIdx);
    TLLM_LOG_DEBUG("storeContextBlocks for request %lu on window %d with %zu unique tokens", llmRequest.mRequestId,
        mWindowSize, uniqueTokens.size());
    if (uniqueTokens.empty())
    {
        return;
    }
    // Respect chunked prefill: getUsableUniqueTokenCountForReuse caps at
    // contextCurrentPosition when prefill is not yet complete, preventing us from
    // storing blocks whose KV state has not been computed yet.
    //
    // Legacy-test fallback: some unit tests (e.g. the VSWA suite ported from PR #12004)
    // call storeContextBlocks immediately after addSequenceBatch without running
    // simulatePrefillCompletion, so contextCurrentPosition is 0 and
    // getUsableUniqueTokenCountForReuse would return 0.  Treating that degenerate state
    // as "prefill complete" by falling back to uniqueTokens.size() - 1 preserves the
    // pre-#13029 single-shot test semantics without affecting production (production
    // always sets contextCurrentPosition > 0 before invoking storeContextBlocks).
    SizeType32 usableUniqueTokenCount = getUsableUniqueTokenCountForReuse(uniqueTokens, llmRequest);
    if (usableUniqueTokenCount == 0 && llmRequest.getContextCurrentPosition() == 0)
    {
        usableUniqueTokenCount = static_cast<SizeType32>(uniqueTokens.size()) - 1;
    }
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableUniqueTokenCount, getTokensPerBlock(), false);
    if (blockedUniqueTokens.empty())
    {
        return;
    }
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);

    // Convert beam-0 cache block IDs to BlockPtrs. Placeholder slots are represented
    // by blocks whose isPlaceholder() is true; storeBlocks handles those specially.
    auto const& beamBlockIds = cacheBlockIds[beamIdx];
    std::vector<BlockPtr> beam0Blocks;
    beam0Blocks.reserve(std::min<std::size_t>(beamBlockIds.size(), blockKeys.size()));
    for (std::size_t i = 0; i < beamBlockIds.size() && i < blockKeys.size(); ++i)
    {
        auto block = getBlockById(beamBlockIds[i]);
        if (!block)
        {
            break;
        }
        beam0Blocks.push_back(std::move(block));
    }
    blockKeys.resize(beam0Blocks.size());
    (void) storeBlocks(std::move(blockKeys), beam0Blocks);
}

void WindowBlockManager::createBlockScalePools(SizeType32 quantBlockSize)
{
    SizeType32 const numEltsPerContainer = getNumEltsPerContainer();
    SizeType32 numPools = mPools.size();
    for (SizeType32 i = 0; i < numPools; ++i)
    {
        auto& kvPool = mPools[i];
        if (kvPool.containsIndexerKCache || kvPool.containsBlockScales)
        {
            continue;
        }
        TLLM_CHECK_WITH_INFO((kvPool.sizePerHead * numEltsPerContainer) % quantBlockSize == 0,
            "Cannot use FP4 quantization since kvPool.sizePerHead is not divisible by FP4 quantBlockSize.");
        auto blockScaleSizePerHead = kvPool.sizePerHead * numEltsPerContainer / quantBlockSize;
        mPools.emplace_back(kvPool.numLayers, kvPool.kvFactor, kvPool.numKvHeads, blockScaleSizePerHead,
            kvPool.tokensPerBlock,
            /*primaryPool=*/nullptr,
            /*secondaryPool=*/nullptr,
            /*containsBlockScales=*/true,
            /*containsIndexerKCache=*/false);
    }
}

void WindowBlockManager::createIndexerKCachePools()
{
    SizeType32 numPools = mPools.size();
    for (SizeType32 i = 0; i < numPools; ++i)
    {
        auto& kvPool = mPools[i];
        if (kvPool.containsIndexerKCache || kvPool.containsBlockScales)
        {
            continue;
        }
        // scaleSize evaluates to 4 at indexHeadDim=128 / quantBlockSize=128 for both
        // FP8 (float32 scale) and FP4 (packed UE8M0 x4 stored as one int32). The
        // data size is either indexHeadDim bytes (one FP8 byte per element) or
        // indexHeadDim / 2 bytes (two packed FP4 E2M1 codes per byte).
        SizeType32 scaleSize = mIndexerKCacheIndexHeadDim / mIndexerKCacheQuantBlockSize * 4;
        SizeType32 dataSize = mIndexerKCacheUseFp4 ? mIndexerKCacheIndexHeadDim / 2 : mIndexerKCacheIndexHeadDim;
        SizeType32 perTokenSize = scaleSize + dataSize;

        mPools.emplace_back(kvPool.numLayers, kvPool.kvFactor, 1, perTokenSize, kvPool.tokensPerBlock,
            /*primaryPool=*/nullptr,
            /*secondaryPool=*/nullptr,
            /*containsBlockScales=*/false,
            /*containsIndexerKCache=*/true);
    }
}

void BlockManager::allocatePools(bool useUvm)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.allocatePools(useUvm);
    }
}

void WindowBlockManager::allocatePools(bool useUvm)
{
    constexpr nvinfer1::DataType kScaleDtypeNVFP4 = nvinfer1::DataType::kFP8;

    // Allocate a memory pool backing the blocks for each numKvHeads
    // TODO(oargov): allocate pools in a single buffer and split it, to avoid fragmentation
    for (auto& pool : mPools)
    {
        auto blockSize = pool.blockSize;
        auto poolDtype = pool.containsBlockScales ? kScaleDtypeNVFP4 : mDataType;
#ifdef ENABLE_FP4
        auto const poolIsFP4 = poolDtype == nvinfer1::DataType::kFP4;
#else
        auto const poolIsFP4 = false;
#endif

        if (poolIsFP4)
        {
            poolDtype = nvinfer1::DataType::kINT8;
        }
        if (pool.containsIndexerKCache)
        {
            poolDtype = nvinfer1::DataType::kUINT8;
        }

        nvinfer1::Dims cacheShape = isRecurrentState()
            ? ITensor::makeShape({pool.numLayers, mNumPrimaryBlocks, mKVFactor, blockSize})
            : ITensor::makeShape({mNumPrimaryBlocks, pool.numLayers, mKVFactor, blockSize});
        pool.layerFirstLayout = isRecurrentState();

        TLLM_LOG_DEBUG(
            "[%s] Allocating primary pool with %d blocks for %d layers with %d kv heads, shape={%d, %d, %d, %d}%s",
            mLogPrefix.c_str(), mNumPrimaryBlocks, pool.numLayers, pool.numKvHeads, cacheShape.d[0], cacheShape.d[1],
            cacheShape.d[2], cacheShape.d[3], pool.layerFirstLayout ? " (layer-first)" : "");

        if (useUvm)
            pool.primaryPtr = BufferManager::managed(cacheShape, poolDtype);
        else
            pool.primaryPtr = mBufferManager.gpuSync(cacheShape, poolDtype);
        if (mNumSecondaryBlocks > 0)
        {
            nvinfer1::Dims cacheShapeOffload = isRecurrentState()
                ? ITensor::makeShape({pool.numLayers, mNumSecondaryBlocks, mKVFactor, blockSize})
                : ITensor::makeShape({mNumSecondaryBlocks, pool.numLayers, mKVFactor, blockSize});
            TLLM_LOG_DEBUG("[%s] Allocating secondary pool with %d blocks for %d layers with %d kv heads",
                mLogPrefix.c_str(), mNumSecondaryBlocks, pool.numLayers, pool.numKvHeads);
            pool.secondaryPtr = BufferManager::pinned(cacheShapeOffload, poolDtype);
        }
    }
}

void BlockManager::releasePools()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.releasePools();
    }
}

void WindowBlockManager::releasePools()
{
    for (auto& pool : mPools)
    {
        if (pool.primaryPtr)
        {
            pool.primaryPtr->release();
        }
        if (pool.secondaryPtr)
        {
            pool.secondaryPtr->release();
        }
    }
    mBufferManager.getStream().synchronize();
    mBufferManager.memoryPoolTrimTo(0);
}

void BlockManager::startScheduling()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.startScheduling();
    }
}

void WindowBlockManager::startScheduling()
{
    mSchedulingNumFreeBlocks = mEvictionPolicy->getNumFreeBlocks(kPrimaryLevel);
    for (auto& [requestId, slotAllocatedBlocks] : mAllocatedBlocksPerSeq)
    {
        for (auto& allocatedBlock : slotAllocatedBlocks)
        {
            allocatedBlock->startScheduling();
        }
    }
}

void WindowBlockManager::freeLeafBlock(BlockPtr const& block)
{
    // The eviction policy needs blocks to still be linked to their old parents when they're reclaimed.
    // This is so it can check if the parent should be queued for eviction.
    block->freeLeafBlock();
}

BlockPtr WindowBlockManager::getFreeBlock(GenerationRequest& sequence, executor::RetentionPriority priority,
    std::optional<std::chrono::milliseconds> durationMs, executor::KvCacheTransferMode mode,
    std::string const& directory, bool wantPlaceholder)
{
    // eviction policy get free primary block
    auto [block, canOffload] = mEvictionPolicy->getFreeBlock(kPrimaryLevel, wantPlaceholder);
    if (block->getUniqueTokens().empty())
    {
        ++mAllocNewBlocks;
    }
    ++mAllocTotalBlocks;
    // Offloading is an option only when these conditions are met:
    // 1. Block contains state (evidenced by presence of tokens)
    // 2. Eviction policy indicated block can be offloaded
    // 3. At least one free block in secondary memory
    if (!wantPlaceholder && !block->getUniqueTokens().empty() && canOffload
        && mEvictionPolicy->getNumFreeBlocks(kSecondaryLevel) > 0)
    {
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));

        // Claim both blocks BEFORE the swap so that getCacheLevel() returns the
        // correct pre-swap level.  Previously the claims happened after the swap,
        // causing claimBlock to erase from the wrong per-level free queue (UB) and
        // to modify the wrong level's free-block counter — the root cause of
        // getNumFreeBlocks() exceeding getMaxNumBlocks() in disagg/prefill mode
        // (GitHub #11879).
        mEvictionPolicy->claimBlock(block);        // primary block → claimed from primary queue
        mEvictionPolicy->claimBlock(offloadBlock); // secondary block → claimed from secondary queue

        mTransferManager->offload(block, offloadBlock, mPools, 0, mode, directory);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel),
                mWindowSize);
        }
        // Release block (now secondary after swap) into secondary block queue,
        // preserving its existing priority.
        mEvictionPolicy->releaseBlock(block);
        // offloadBlock (now primary after swap) is already claimed above.
        // The final claimBlock() below will be a no-op for the queue but still
        // applies the caller's priority/durationMs.
        block = offloadBlock;
    }

    // True priority eviction: detach ONLY this block from the lookup tree.
    // Descendants remain in the tree and free queue, where they can be evicted
    // independently according to their own priority.  Previously, freeChildren()
    // also detached all descendants, which prevented high-priority interior blocks
    // from surviving eviction pressure on their low-priority leaf children.
    //
    // Serialize with the lookup tree mutex: storeBlocks, analyzePrefixReuse,
    // and addSequenceBatch all hold this mutex while accessing the trie, so
    // detachFromLookupNode must do the same.
    {
        std::lock_guard<std::recursive_mutex> treeLock(mLookupTree->getMutex());
        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueRemovedEvent(block, mWindowSize);
        }
        block->detachFromLookupNode();
    }
    // Claim the block in primary block queue
    mEvictionPolicy->claimBlock(block, priority, durationMs);
    TLLM_LOG_DEBUG("%s::getFreeBlock - Block %d is now acquired by sequence %d", mLogPrefix.c_str(),
        block->getBlockId(), sequence.getRequestId());

    return block;
}

void WindowBlockManager::setOffsets(tk::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape,
    SizeType32 beamIdx, SizeType32 blockIdx, KVCacheBlock::IdType blockId) const
{
    auto constexpr kIdx = 0;
    auto constexpr vIdx = 1;

    auto const& block = getBlockById(blockId);
    bool const isSwaPlaceholder = blockId == KVCacheBlock::kPlaceholderBlockId;
    TLLM_CHECK_WITH_INFO(block != nullptr || isSwaPlaceholder, "Unknown KV cache block id %d", blockId);
    for (SizeType32 poolIdx = 0; poolIdx < static_cast<SizeType32>(mPools.size()); poolIdx++)
    {
        auto const& pool = mPools.at(poolIdx);
        for (auto const xIdx : {kIdx, vIdx})
        {
            auto constexpr layerIdx = 0;
            auto const offsetIndex = tensorrt_llm::common::flat_index(offsetsShape.d, poolIdx, beamIdx, xIdx, blockIdx);
            auto const fieldIdx = (mCacheType == CacheType::kSELFKONLY || isRecurrentState()) ? 0 : xIdx;
            auto const blockIndex = [&]() -> tk::KVCacheIndex
            {
                if (isSwaPlaceholder || block->isPlaceholder())
                {
                    return tk::KVCacheIndex::nullIndex;
                }
                if (pool.layerFirstLayout)
                {
                    // Layer-first layout: {numLayers, numBlocks, kvFactor, blockSize}
                    // Flat index: layerIdx * numBlocks * kvFactor + blockIdx * kvFactor + fieldIdx
                    return tk::KVCacheIndex{common::flat_index3(
                        layerIdx, block->getMemoryPoolBlockIndex(), fieldIdx, mNumPrimaryBlocks, mKVFactor)};
                }
                return tk::KVCacheIndex{common::flat_index3(
                    block->getMemoryPoolBlockIndex(), layerIdx, fieldIdx, pool.numLayers, mKVFactor)};
            }();
            offsetsPtr[offsetIndex] = blockIndex;
        }
    }
}

void BlockManager::setOffsets(tk::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
    SizeType32 blockIdx, KVCacheBlock::IdType blockId, SizeType32 windowSize) const
{
    mWindowBlockManagers.at(windowSize).setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
}

void BlockManager::onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock, SizeType32 windowSize,
    executor::KvCacheTransferMode mode, std::string const& directory)
{
    mWindowBlockManagers.at(windowSize).onboardBlock(sequence, offloadBlock, mode, directory);
}

void WindowBlockManager::onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock,
    executor::KvCacheTransferMode mode, std::string const& directory)
{
    if (!offloadBlock->isPlaceholder() && !offloadBlock->isPrimary())
    {
        auto block = getFreeBlock(
            sequence, executor::KvCacheRetentionConfig::kDefaultRetentionPriority, std::nullopt, mode, directory);
        mTransferManager->onboard(offloadBlock, block, mPools, 0, mode, directory);
        // swap linear block offsets (i.e. make block the offload block and vice versa)
        offloadBlock->swapMemoryPoolBlockOffset(block);

        if (mEventManager)
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(offloadBlock->getHash()).cacheLevelUpdated(kSecondaryLevel, kPrimaryLevel),
                mWindowSize);
        }
        mEvictionPolicy->releaseBlock(block); // append block to offload queue
                                              // offloadBlock is now in primary memory pool
    }
}

void BlockManager::offloadBlock(
    BlockPtr const& block, SizeType32 windowSize, executor::KvCacheTransferMode mode, std::string const& directory)
{
    mWindowBlockManagers.at(windowSize).offloadBlock(block, mode, directory);
}

void WindowBlockManager::offloadBlock(
    BlockPtr const& block, executor::KvCacheTransferMode mode, std::string const& directory)
{
    // The current default behavior is to offload the out-of-window block
    // to secondary block pool to allow more free primary blocks for reuse.
    // However, such behavior does not take account whether the offloaded
    // block is useful or not and may just lead to more traffic instead.
    // The ideal way of this is to dedicate the offloading of the block
    // to the eviction policy.
    if (!block->isPlaceholder() && block->isPrimary())
    {
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        // If we're swapping a block to secondary memory, maintain the prior priority values.
        mEvictionPolicy->claimBlock(offloadBlock);
        mTransferManager->offload(block, offloadBlock, mPools, 0, mode, directory);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel),
                mWindowSize);
        }
        mEvictionPolicy->releaseBlock(offloadBlock); // append offloadBlock to mFreePrimaryBlocks queue
                                                     // block is now in secondary memory
    }
}

PrefixReuseSummary BlockManager::analyzePrefixReuse(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    TLLM_CHECK_WITH_INFO(!isVariableWindow(), "analyzePrefixReuse does not work for variable window attention");
    auto const& onlyManager = mWindowBlockManagers.cbegin()->second;
    return onlyManager.analyzePrefixReuse(uniqueTokens, llmRequest);
}

PrefixReuseSummary WindowBlockManager::analyzePrefixReuse(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size(), mTokensPerBlock, false);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);

    PrefixReuseSummary summary;

    std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());
    auto reuseMatches = findReusableBlockMatches(
        blockKeys, /*enablePartialReuse=*/false, /*copyOnPartialReuse=*/false, std::numeric_limits<SizeType32>::max());

    for (auto const& match : reuseMatches.matches)
    {
        if (match.isTraversalOnly)
        {
            continue;
        }

        ++summary.reusableBlocksAll;
        if (match.block->hasRefs())
        {
            ++summary.reusableBlocksAllocated;
        }
    }
    summary.firstNewBlock = reuseMatches.firstNewBlock;

    TLLM_LOG_DEBUG("%s::analyzePrefixReuse - reusableAllocated=%d, reusableAll=%d, hasNewBlock=%d", mLogPrefix.c_str(),
        summary.reusableBlocksAllocated, summary.reusableBlocksAll, summary.firstNewBlock.has_value());
    return summary;
}

WindowBlockManager::ReuseMatchResult WindowBlockManager::findReusableBlockMatches(
    std::vector<BlockKey> const& blockKeys, bool enablePartialReuse, bool copyOnPartialReuse,
    SizeType32 maxMatchedTokens) const
{
    ReuseMatchResult result;
    std::vector<ReuseMatch> candidateMatches;
    candidateMatches.reserve(blockKeys.size());
    auto searchNode = mCachedBlocksRoot ? mCachedBlocksRoot->getLookupNode() : nullptr;
    SizeType32 candidateMatchedTokens{0};
    SizeType32 latestMissingAnchorEndToken{0};

    auto updateSafePrefix = [&]()
    {
        if (!mIsSWA || latestMissingAnchorEndToken == 0
            || candidateMatchedTokens >= latestMissingAnchorEndToken + mWindowSize)
        {
            result.matches = candidateMatches;
            result.totalMatchedTokens = candidateMatchedTokens;
        }
    };

    for (auto const& blockKey : blockKeys)
    {
        if (!searchNode || blockKey.uniqueTokens.empty())
        {
            break;
        }

        auto exactMatch = searchNode->findMatchingNode(blockKey);
        if (exactMatch.has_value())
        {
            auto const numMatchedTokens = static_cast<SizeType32>(blockKey.uniqueTokens.size());
            if (candidateMatchedTokens + numMatchedTokens > maxMatchedTokens)
            {
                break;
            }

            auto const node = exactMatch->node;
            auto existing = node->getValue(mWindowSize);
            candidateMatchedTokens += numMatchedTokens;

            if (existing.has_value() && *existing)
            {
                auto block = *existing;
                candidateMatches.push_back(ReuseMatch{block, numMatchedTokens, !block->isFull(), false});
            }
            else if (mIsSWA)
            {
                candidateMatches.push_back(ReuseMatch{nullptr, numMatchedTokens, false, true});
                latestMissingAnchorEndToken = std::max(latestMissingAnchorEndToken, candidateMatchedTokens);
            }
            else
            {
                break;
            }

            searchNode = node;
            updateSafePrefix();
            continue;
        }

        if (enablePartialReuse)
        {
            auto partialMatches = searchNode->findPartiallyMatchingNodes(blockKey);
            for (auto const& match : partialMatches)
            {
                auto existing = match.node->getValue(mWindowSize);
                if (!existing.has_value() || !(*existing))
                {
                    continue;
                }

                auto block = *existing;
                if (copyOnPartialReuse || (!block->hasRefs() && block->isLeaf()))
                {
                    auto const numMatchedTokens = static_cast<SizeType32>(match.key.uniqueTokens.size());
                    if (candidateMatchedTokens + numMatchedTokens <= maxMatchedTokens)
                    {
                        candidateMatchedTokens += numMatchedTokens;
                        candidateMatches.push_back(ReuseMatch{block, numMatchedTokens, true, false});
                        updateSafePrefix();
                    }
                    break;
                }
            }
        }
        break;
    }

    if (result.matches.size() < blockKeys.size())
    {
        result.firstNewBlock = blockKeys[result.matches.size()];
    }
    return result;
}

WindowBlockManager::ClaimResult WindowBlockManager::claimMatchingBlocks(GenerationRequest& sequence,
    SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest, size_t requestIdx,
    PartialClaimTracker& tracker, std::vector<ClaimResult>& claimResults)
{
    // NOTE: Caller must hold mLookupTree->getMutex().
    TLLM_CHECK_WITH_INFO(!(isRecurrentState()) || inputLength == llmRequest.getPromptLen(),
        "Recurrent state does not support CP or truncation yet.");

    ClaimResult result;
    result.numContextBlocks = numContextBlocks;

    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    // Prepare block keys — guard for cross-KV without encoder tokens (e.g., Whisper).
    auto constexpr beamIdx = 0;
    bool const isSelfCache = mCacheType == CacheType::kSELF || mCacheType == CacheType::kSELFKONLY;
    bool const hasUniqueTokens = isSelfCache
        || (llmRequest.getEncoderUniqueTokens().has_value() && llmRequest.getEncoderUniqueTokens().value());

    if (hasUniqueTokens)
    {
        auto const& uniqueTokens
            = isSelfCache ? llmRequest.getUniqueTokens(beamIdx) : *(llmRequest.getEncoderUniqueTokens().value());

        auto blockedUniqueTokens
            = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, inputLength - 1, mTokensPerBlock, true);
        if (inputLength % mTokensPerBlock == 1)
        {
            blockedUniqueTokens.emplace_back();
        }

        result.blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
    }

    auto config = llmRequest.getKvCacheRetentionConfig();
    result.perBlockRetentions = config.value_or(executor::KvCacheRetentionConfig())
                                    .getPerBlockRetentionPriorityDuration(mTokensPerBlock, inputLength);
    result.mode = config.value_or(executor::KvCacheRetentionConfig()).getTransferMode();
    result.directory = config.value_or(executor::KvCacheRetentionConfig()).getDirectory();

    if (result.mode != executor::KvCacheTransferMode::DRAM && result.directory.empty())
    {
        TLLM_LOG_WARNING(
            "Transfer mode %d specified without directory, falling back to DRAM mode", static_cast<int>(result.mode));
        result.mode = executor::KvCacheTransferMode::DRAM;
    }

    TLLM_CHECK(result.perBlockRetentions.size() == static_cast<size_t>(numContextBlocks));

    // Phase 1: Walk radix tree, claim matching blocks — no onboard, no getFreeBlock
    // NOTE: Caller must hold mLookupTree->getMutex().

    // Compute shareLastContextBlockAmongBeams for the batch-add allocation path.
    auto const beamWidth = sequence.getBeamWidth();
    bool const isShareLastContextBlock = mCacheType == CacheType::kCROSS || inputLength % mTokensPerBlock == 0;
    result.numSharedContextBlocks
        = (beamWidth > 1 && !isShareLastContextBlock) ? numContextBlocks - 1 : numContextBlocks;
    result.shareLastContextBlockAmongBeams = result.numSharedContextBlocks == numContextBlocks;
    auto reuseMatches = findReusableBlockMatches(
        result.blockKeys, mEnablePartialReuse, mCopyOnPartialReuse, sequence.getCurrentPrepopulatedPromptLen());
    result.totalMatchedTokens = reuseMatches.totalMatchedTokens;

    for (int bi = 0; bi < result.numSharedContextBlocks && bi < static_cast<int>(reuseMatches.matches.size()); ++bi)
    {
        auto const& match = reuseMatches.matches[bi];
        if (isRecurrentState())
        {
            TLLM_CHECK(match.isPartialMatch == false);
        }

        ClaimResult::ClaimedBlock claimed;
        claimed.numMatchedTokens = match.numMatchedTokens;
        claimed.isPartialMatch = match.isPartialMatch;
        claimed.needsCopy = false;
        claimed.isTraversalOnly = match.isTraversalOnly;
        if (match.isTraversalOnly)
        {
            claimed.block = KVCacheBlock::createPlaceholder();
            claimed.isPlaceholder = true;
            result.claimedBlocks.push_back(std::move(claimed));
            continue;
        }

        auto matchingBlock = match.block;
        auto const partialMatch = match.isPartialMatch;
        claimed.block = matchingBlock;
        claimed.isPlaceholder = matchingBlock->isPlaceholder();
        if (!claimed.isPlaceholder)
        {
            result.latestMatchingNonPlaceholderBlockIdx = bi;
        }

        // Priority update event
        if (result.perBlockRetentions[bi].retentionPriority.has_value()
            && matchingBlock->getPriority() != result.perBlockRetentions[bi].retentionPriority && mEventManager)
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(matchingBlock->getHash())
                    .priorityUpdated(matchingBlock->getPriority(), *result.perBlockRetentions[bi].retentionPriority),
                mWindowSize);
        }

        if (partialMatch)
        {
            if (matchingBlock->hasRefs() || !matchingBlock->isLeaf())
            {
                // Block in use or has children — always needs copy.
                claimed.needsCopy = true;
                if (!matchingBlock->hasRefs())
                {
                    // Unreferenced non-leaf: claim to protect from eviction during copies.
                    // Use tracker to assign release responsibility to the last copier.
                    mEvictionPolicy->claimBlock(matchingBlock, result.perBlockRetentions[bi].retentionPriority,
                        result.perBlockRetentions[bi].durationMs);

                    auto const blockId = matchingBlock->getBlockId();
                    auto tIt = tracker.map.find(blockId);
                    if (tIt != tracker.map.end())
                    {
                        if (tIt->second.fullyMatched)
                        {
                            // A full match holds this block — do not release.
                            claimed.shouldReleaseCopySource = false;
                        }
                        else
                        {
                            // Previous copier no longer responsible for release.
                            claimResults[tIt->second.requestIdx]
                                .claimedBlocks[tIt->second.claimedIdx]
                                .shouldReleaseCopySource
                                = false;
                            claimed.shouldReleaseCopySource = true;
                        }
                        tIt->second.requestIdx = requestIdx;
                        tIt->second.claimedIdx = result.claimedBlocks.size();
                    }
                    else
                    {
                        tracker.map[blockId] = {requestIdx, result.claimedBlocks.size(), /*fullyMatched=*/false};
                        claimed.shouldReleaseCopySource = true;
                    }
                }
            }
            else
            {
                // Leaf with no refs — decide reuse vs copy using the batch tracker.
                // Do NOT call freeLeafBlock here (cascade prune would corrupt the trie
                // for later requests). Claim to protect from eviction; freeLeafBlock is
                // deferred to Phase 2 for the single reuser.
                auto const blockId = matchingBlock->getBlockId();
                auto tIt = tracker.map.find(blockId);
                if (tIt != tracker.map.end())
                {
                    if (tIt->second.fullyMatched)
                    {
                        // A previous request already fully matched this block — must copy.
                        claimed.needsCopy = true;
                    }
                    else
                    {
                        // A previous request was going to reuse — bump it to copy,
                        // and this request becomes the new reuser.
                        claimResults[tIt->second.requestIdx].claimedBlocks[tIt->second.claimedIdx].needsCopy = true;
                        claimed.needsCopy = false;
                        tIt->second.requestIdx = requestIdx;
                        tIt->second.claimedIdx = result.claimedBlocks.size();
                    }
                }
                else
                {
                    // First request to partially match this leaf — reuse it.
                    claimed.needsCopy = false;
                    tracker.map[blockId] = {requestIdx, result.claimedBlocks.size(), /*fullyMatched=*/false};
                }
                mEvictionPolicy->claimBlock(matchingBlock, result.perBlockRetentions[bi].retentionPriority,
                    result.perBlockRetentions[bi].durationMs);
            }
        }
        else
        {
            // Full match — claim block (removes from free queue, protecting from eviction)
            mEvictionPolicy->claimBlock(matchingBlock, result.perBlockRetentions[bi].retentionPriority,
                result.perBlockRetentions[bi].durationMs);

            // If a previous request was going to reuse or release this block via partial match,
            // it must now copy instead — a full match takes priority.
            {
                auto const blockId = matchingBlock->getBlockId();
                auto tIt = tracker.map.find(blockId);
                if (tIt != tracker.map.end() && !tIt->second.fullyMatched)
                {
                    claimResults[tIt->second.requestIdx].claimedBlocks[tIt->second.claimedIdx].needsCopy = true;
                    claimResults[tIt->second.requestIdx].claimedBlocks[tIt->second.claimedIdx].shouldReleaseCopySource
                        = false;
                    tIt->second.fullyMatched = true;
                }
                else
                {
                    tracker.map[blockId] = {requestIdx, result.claimedBlocks.size(), /*fullyMatched=*/true};
                }
            }
        }

        result.claimedBlocks.push_back(std::move(claimed));
    }

    TLLM_LOG_DEBUG("%s::claimMatchingBlocks for request %lu - Claimed %zu blocks, %d matched tokens",
        mLogPrefix.c_str(), sequence.getRequestId(), result.claimedBlocks.size(), result.totalMatchedTokens);

    return result;
}

SizeType32 WindowBlockManager::onboardAndAllocateBlocks(
    GenerationRequest& sequence, LlmRequest& llmRequest, ClaimResult& claimResult, bool isEnableBlockReuse)
{
    // NOTE: Caller must hold mLookupTree->getMutex().
    std::set<KVCacheBlock::IdType> reusedBlockIds;
    auto blockItr = claimResult.blockKeys.begin();
    SizeType32 bi = 0;

    // Process claimed (matched) blocks: onboard + addBlockToAllBeams
    for (auto& claimed : claimResult.claimedBlocks)
    {
        if (claimed.isTraversalOnly)
        {
            TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks for request %lu - Traversed missing SWA anchor",
                mLogPrefix.c_str(), sequence.getRequestId());
            addBlockToAllBeams(claimed.block, sequence);
            ++blockItr;
            ++bi;
            continue;
        }

        KVCacheBlock::IdType matchingBlockId = claimed.block->getBlockId();

        if (claimed.isPartialMatch && claimed.needsCopy)
        {
            // Partial match needing copy: allocate new block, copy from source, use new block
            auto copySource = claimed.block;
            auto newBlock = getFreeBlock(sequence, copySource->getPriority(), copySource->getDurationMs(),
                claimResult.mode, claimResult.directory);
            mTransferManager->onboard(
                copySource, newBlock, mPools, claimed.numMatchedTokens, claimResult.mode, claimResult.directory);
            // Release the claimed non-leaf copy source back to the free queue now that
            // the copy is done. The tracker ensures only the last copier releases.
            if (claimed.shouldReleaseCopySource && !copySource->hasRefs())
            {
                mEvictionPolicy->releaseBlock(copySource);
            }
            claimed.block = newBlock;
            if (blockItr != claimResult.blockKeys.end())
            {
                claimed.block->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
            }
            claimed.block->setHash();
            TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks for request %lu - Copied partially filled block %d",
                mLogPrefix.c_str(), sequence.getRequestId(), matchingBlockId);
        }
        else if (claimed.isPartialMatch)
        {
            // Partial leaf match — this request is the sole reuser (Phase 1 tracker chose it).
            // Detach from the trie now so that storeContextBlocks can re-insert cleanly.
            // Cascade pruning is safe here because all ancestor blocks along the matched
            // path were claimed in Phase 1 (their trie values are intact, stopping cascade).
            freeLeafBlock(claimed.block);
            TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks for request %lu - Reused partially filled block %d",
                mLogPrefix.c_str(), sequence.getRequestId(), matchingBlockId);
        }
        else
        {
            // Full match — already claimed in Phase 1
            TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks for request %lu - Matched full block %d", mLogPrefix.c_str(),
                sequence.getRequestId(), matchingBlockId);
        }

        onboardBlock(sequence, claimed.block, claimResult.mode, claimResult.directory);
        addBlockToAllBeams(claimed.block, sequence);
        if (!claimed.isPlaceholder)
        {
            ++mReusedBlocks;
            if (claimed.isPartialMatch)
            {
                ++mPartialReusedBlocks;
            }
            else
            {
                ++mFullReusedBlocks;
            }
            if (!reusedBlockIds.count(matchingBlockId))
            {
                reusedBlockIds.insert(matchingBlockId);
                ++mReusedUniqueBlocks;
            }
        }
        ++blockItr;
        ++bi;
    }

    // Allocate non-matching shared context blocks
    for (; bi < claimResult.numSharedContextBlocks; ++bi)
    {
        bool shouldAllocate = true;
        if (isRecurrentState())
        {
            if (isEnableBlockReuse)
            {
                shouldAllocate = mLinearAttentionMetadata->shouldAllocateRecurrentStates(
                    /*currentBlockEndTokenIdx=*/(bi + 1) * mTokensPerBlock, llmRequest.getPromptLen(), mTokensPerBlock);
            }
            else
            {
                // When block reuse is disabled, only the last context block needs real memory
                // to store the current recurrent state. All other blocks are placeholders.
                shouldAllocate = (bi == claimResult.numContextBlocks - 1);
            }
            TLLM_LOG_DEBUG(
                "%s::onboardAndAllocateBlocks - Recurrent state block %d. shouldAllocate=%d for sequence %lu",
                mLogPrefix.c_str(), bi, shouldAllocate, sequence.getRequestId());
        }

        auto freeBlock = getFreeBlock(sequence,
            claimResult.perBlockRetentions[bi].retentionPriority.value_or(
                executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
            claimResult.perBlockRetentions[bi].durationMs, claimResult.mode, claimResult.directory,
            /*wantPlaceholder=*/!shouldAllocate);
        addBlockToAllBeams(freeBlock, sequence);
        TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks - No match, allocated new block %d for sequence %lu",
            mLogPrefix.c_str(), freeBlock->getBlockId(), sequence.getRequestId());
        if (blockItr != claimResult.blockKeys.end())
        {
            freeBlock->setBlockKey(*blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
            ++blockItr;
        }
        freeBlock->setHash();
        ++mMissedBlocks;
    }

    // Allocate non-shared last blocks (beam > 1)
    auto const beamWidth = sequence.getBeamWidth();
    for (int nbi = claimResult.numSharedContextBlocks; nbi < claimResult.numContextBlocks; ++nbi)
    {
        for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            auto freeBlock = getFreeBlock(sequence,
                claimResult.perBlockRetentions[nbi].retentionPriority.value_or(
                    executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                claimResult.perBlockRetentions[nbi].durationMs, claimResult.mode, claimResult.directory);
            addBlockToBeam(freeBlock, sequence, beamIdx);
            if (blockItr != claimResult.blockKeys.end())
            {
                freeBlock->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                ++blockItr;
            }
            freeBlock->setHash();
            TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks - Beam %d. Allocated non-shared block %d for bi %d",
                mLogPrefix.c_str(), beamIdx, freeBlock->getBlockId(), nbi);
        }
        ++mMissedBlocks;
        if (blockItr != claimResult.blockKeys.end())
        {
            ++blockItr;
        }
    }

    // Finalize matched token count (purge trailing placeholders for recurrent states)
    auto numMatchedTokens = claimResult.totalMatchedTokens;
    if (isRecurrentState())
    {
        numMatchedTokens = (claimResult.latestMatchingNonPlaceholderBlockIdx + 1) * mTokensPerBlock;
    }
    sequence.setCurrentPrepopulatedPromptLen(numMatchedTokens);

    // Update stats and return prepopulated length
    mReusedTokens += static_cast<double>(numMatchedTokens);
    bool const isSelfCache = mCacheType == CacheType::kSELF || mCacheType == CacheType::kSELFKONLY;
    bool const hasUniqueTokens = isSelfCache
        || (llmRequest.getEncoderUniqueTokens().has_value() && llmRequest.getEncoderUniqueTokens().value());
    if (hasUniqueTokens)
    {
        auto constexpr beamIdx = 0;
        auto const& uniqueTokens
            = isSelfCache ? llmRequest.getUniqueTokens(beamIdx) : *(llmRequest.getEncoderUniqueTokens().value());
        mTotalInputTokens += static_cast<double>(uniqueTokens.size());
    }
    else
    {
        mTotalInputTokens += static_cast<double>(claimResult.numContextBlocks * mTokensPerBlock);
    }

    SizeType32 numConnectorMatchedTokens = 0;
    if (mKvCacheConnectorManager && !llmRequest.isDummyRequest())
    {
        numConnectorMatchedTokens
            = mKvCacheConnectorManager->getNumNewMatchedTokens(llmRequest, sequence.getCurrentPrepopulatedPromptLen());
    }

    auto const totalPrepopulatedLen = sequence.getCurrentPrepopulatedPromptLen() + numConnectorMatchedTokens;
    TLLM_LOG_DEBUG("%s::onboardAndAllocateBlocks: Request %lu, prepopulatedPromptLen %d, numConnectorMatchedTokens %d",
        mLogPrefix.c_str(), llmRequest.mRequestId, sequence.getCurrentPrepopulatedPromptLen(),
        numConnectorMatchedTokens);
    return totalPrepopulatedLen;
}

WindowBlockManager::ClaimResult WindowBlockManager::buildClaimResultMetadata(
    GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest)
{
    // Build ClaimResult metadata without walking the radix tree.
    // Used for non-reuse path where all blocks are freshly allocated.
    ClaimResult result;
    result.numContextBlocks = numContextBlocks;

    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    // Prepare block keys (needed by onboardAndAllocateBlocks for setBlockKey/setHash)
    bool const isSelfCache = mCacheType == CacheType::kSELF || mCacheType == CacheType::kSELFKONLY;
    bool const hasUniqueTokens = isSelfCache
        || (llmRequest.getEncoderUniqueTokens().has_value() && llmRequest.getEncoderUniqueTokens().value());

    if (hasUniqueTokens)
    {
        auto constexpr beamIdx = 0;
        auto const& uniqueTokens
            = isSelfCache ? llmRequest.getUniqueTokens(beamIdx) : *(llmRequest.getEncoderUniqueTokens().value());

        auto blockedUniqueTokens
            = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, inputLength - 1, mTokensPerBlock, true);
        if (inputLength % mTokensPerBlock == 1)
        {
            blockedUniqueTokens.emplace_back();
        }
        result.blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
    }

    auto config = llmRequest.getKvCacheRetentionConfig();
    result.perBlockRetentions = config.value_or(executor::KvCacheRetentionConfig())
                                    .getPerBlockRetentionPriorityDuration(mTokensPerBlock, inputLength);
    result.mode = config.value_or(executor::KvCacheRetentionConfig()).getTransferMode();
    result.directory = config.value_or(executor::KvCacheRetentionConfig()).getDirectory();

    if (result.mode != executor::KvCacheTransferMode::DRAM && result.directory.empty())
    {
        TLLM_LOG_WARNING(
            "Transfer mode %d specified without directory, falling back to DRAM mode", static_cast<int>(result.mode));
        result.mode = executor::KvCacheTransferMode::DRAM;
    }

    TLLM_CHECK(result.perBlockRetentions.size() == static_cast<size_t>(numContextBlocks));

    // Use unified isShareLastContextBlock formula (aligned with PR #10437)
    bool const isShareLastContextBlock = mCacheType == CacheType::kCROSS || inputLength % mTokensPerBlock == 0;
    auto const beamWidth = sequence.getBeamWidth();
    result.numSharedContextBlocks
        = (beamWidth > 1 && !isShareLastContextBlock) ? numContextBlocks - 1 : numContextBlocks;
    result.shareLastContextBlockAmongBeams = result.numSharedContextBlocks == numContextBlocks;

    return result;
}

std::vector<WindowBlockManager::BatchSeqStats> WindowBlockManager::addSequenceBatch(
    std::vector<GenerationRequest*> const& sequences, std::vector<SizeType32> const& inputLengths,
    std::vector<SizeType32> const& numContextBlocksVec,
    std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests, bool isEnableBlockReuse)
{
    auto const n = sequences.size();
    std::vector<ClaimResult> claimResults(n);
    std::vector<BatchSeqStats> results(n);

    // Hold the lock for the entire two-phase operation.
    std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());

    if (isEnableBlockReuse)
    {
        // Phase 1: Claim all matching blocks across all requests.
        // The tracker coordinates partial-match ownership: at most one request reuses a
        // partially-matched leaf block in-place; all others copy.
        PartialClaimTracker tracker;
        for (size_t i = 0; i < n; ++i)
        {
            claimResults[i] = claimMatchingBlocks(
                *sequences[i], inputLengths[i], numContextBlocksVec[i], llmRequests[i].get(), i, tracker, claimResults);
        }
    }
    else
    {
        // Non-reuse: build ClaimResult with metadata only, no radix tree walk.
        // All blocks will be freshly allocated in Phase 2.
        for (size_t i = 0; i < n; ++i)
        {
            claimResults[i] = buildClaimResultMetadata(
                *sequences[i], inputLengths[i], numContextBlocksVec[i], llmRequests[i].get());
        }
    }

    // Phase 2: Onboard + allocate for each request, snapshotting stats between requests.
    for (size_t i = 0; i < n; ++i)
    {
        SizeType32 const preTotalBlocks = mAllocTotalBlocks;
        SizeType32 const preNewBlocks = mAllocNewBlocks;
        SizeType32 const preReused = mReusedBlocks;
        SizeType32 const preMissed = mMissedBlocks;

        results[i].prepopulatedLen
            = onboardAndAllocateBlocks(*sequences[i], llmRequests[i].get(), claimResults[i], isEnableBlockReuse);

        results[i].allocTotalDelta = mAllocTotalBlocks - preTotalBlocks;
        results[i].allocNewDelta = mAllocNewBlocks - preNewBlocks;
        results[i].reusedDelta = mReusedBlocks - preReused;
        results[i].missedDelta = mMissedBlocks - preMissed;
    }

    // No deferred release needed: non-leaf copy sources are released in Phase 2
    // via shouldReleaseCopySource. Leaf copy sources always have refs after Phase 2
    // (held by the reuser or full-matcher), so they would be skipped anyway.

    return results;
}

bool WindowBlockManager::blockInRadixTree(BlockPtr const& block)
{
    return !block->getUniqueTokens().empty() && block->getPrevBlock() != nullptr;
}

std::shared_ptr<KVCacheBlock> WindowBlockManager::findBlocksInReuseTreeByBlockKey(BlockKey const& blockKey)
{
    std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(blockKey.uniqueTokens, blockKey.uniqueTokens.size(), mTokensPerBlock, true);

    std::vector<BlockKey> blockKeys;
    for (auto const& blockedUniqueTokensList : blockedUniqueTokens)
    {
        blockKeys.push_back(blockKey);
        blockKeys.back().uniqueTokens = blockedUniqueTokensList;
    }
    return searchReuseTree(blockKeys);
}

std::shared_ptr<KVCacheBlock> WindowBlockManager::findBlocksInReuseTreeByBlockKeys(
    std::vector<BlockKey> const& blockKeys)
{
    std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());
    return searchReuseTree(blockKeys);
}

std::shared_ptr<KVCacheBlock> WindowBlockManager::searchReuseTree(std::vector<BlockKey> const& blockKeys)
{
    if (blockKeys.empty())
    {
        return mCachedBlocksRoot;
    }

    auto reuseMatches = findReusableBlockMatches(
        blockKeys, /*enablePartialReuse=*/true, /*copyOnPartialReuse=*/true, std::numeric_limits<SizeType32>::max());
    if (reuseMatches.matches.size() != blockKeys.size() || reuseMatches.matches.back().isTraversalOnly)
    {
        return nullptr;
    }
    return reuseMatches.matches.back().block;
}

void BlockManager::syncTransferManagerWithBufferManager()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.syncTransferManagerWithBufferManager();
    }
}

void WindowBlockManager::syncTransferManagerWithBufferManager()
{
    mTransferManager->syncWithBufferManager();
}

void BlockManager::refreshBlocks()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.refreshBlocks();
    }
}

void WindowBlockManager::refreshBlocks()
{
    mEvictionPolicy->refresh();
    mTransferManager->syncTransfers();
}

std::vector<WindowBlockManager::BatchSeqStats> BlockManager::addSequenceBatch(
    std::vector<GenerationRequest*> const& sequences, std::vector<SizeType32> const& inputLengths,
    std::vector<SizeType32> const& numContextBlocksVec,
    std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests, SizeType32 windowSize, bool isEnableBlockReuse)
{
    return mWindowBlockManagers.at(windowSize)
        .addSequenceBatch(sequences, inputLengths, numContextBlocksVec, llmRequests, isEnableBlockReuse);
}

void BlockManager::adjustBlocksIfNeeded(GenerationRequest& sequence)
{
    for (auto& [windowSize, manager] : mWindowBlockManagers)
    {
        mWindowBlockManagers.at(windowSize).adjustBlocksIfNeeded(sequence);
    }
}

void WindowBlockManager::adjustBlocksIfNeeded(GenerationRequest& sequence)
{
    auto const minTokensForBlockDetach = mWindowSize + mTokensPerBlock;
    while (mIsSWA && // A block only go out-of-window in SWA
        (sequence.getNumTokens() - sequence.getNumFrontBlocksRemoved() * getTokensPerBlock()
            >= minTokensForBlockDetach))
    {
        // Detaching block for SWA is non-trivial due to the radix tree structure.
        // For now, when reuse is enabled, we do not detach blocks for SWA.
        detachFrontBlock(sequence);
    }

    if ((sequence.getNumTokens() - 1) % getTokensPerBlock() == 0)
    {
        // Allocating a new block when the last token is a block boundary
        allocateBlock(sequence, /*shareAmongBeams=*/sequence.getBeamWidth() == 1);
        ++mGenAllocBlocks;
        updateLastCacheBlockOffsets(sequence);
    }
}

void WindowBlockManager::addBlockToBeam(BlockPtr const& block, GenerationRequest& sequence, SizeType32 beamIdx)
{
    auto const requestId = sequence.getRequestId();
    block->incRefCount();
    if (sequence.getCacheBlockIds(mWindowSize).at(beamIdx).size() == 0)
    {
        block->setPrevBlockInSeq(nullptr);
    }
    else
    {
        block->setPrevBlockInSeq(mAllocatedBlocksPerSeq.at(requestId).at(
            (sequence.getCacheBlockIds(mWindowSize)[beamIdx].size() - 1) * sequence.getBeamWidth() + beamIdx));
    }
    sequence.addCacheBlock(mWindowSize, beamIdx, block->getBlockId());
    mAllocatedBlocksPerSeq.at(requestId).push_back(block);
}

void WindowBlockManager::addBlockToAllBeams(BlockPtr const& block, GenerationRequest& sequence)
{
    auto const beamWidth = sequence.getBeamWidth();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        addBlockToBeam(block, sequence, beamIdx);
    }
}

void BlockManager::allocateBlock(GenerationRequest& sequence, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).allocateBlock(sequence, false);
}

void BlockManager::copyLinearAttentionBlock(GenerationRequest& sequence, LlmRequest const& llmRequest)
{
    for (auto& [windowSize, manager] : mWindowBlockManagers)
    {
        manager.copyLinearAttentionBlock(sequence, llmRequest);
    }
}

bool WindowBlockManager::tryAllocatePlaceholderForLinearAttention(GenerationRequest& sequence, bool shareAmongBeams)
{
    auto const beamWidth = sequence.getBeamWidth();
    auto const newBlockIdx = sequence.getCacheBlockIds(mWindowSize).at(0).size();
    // The first block is not a placeholder.
    if (newBlockIdx == 0)
    {
        return false;
    }

    // If the last block is saved in lookup tree for reuse, we keep it.
    // A case is that the context seqlen is a multiple of tokens per block, and reuse is enabled.
    int lastBlockId = sequence.getCacheBlockIds(mWindowSize).at(0).back();
    if (getBlockById(lastBlockId)->getLookupNode() != nullptr && mLinearAttentionMetadata->saveLastSnapshot)
    {
        TLLM_LOG_DEBUG(
            "tryAllocatePlaceholderForLinearAttention: corner case to allocate block at generation phase, "
            "lastBlockId=%d, requestId=%lu, numTokens=%d",
            lastBlockId, sequence.getRequestId(), sequence.getNumTokens());
        return false;
    }

    bool isLastBlockSharedAmongBeams = true;
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        if (lastBlockId != sequence.getCacheBlockIds(mWindowSize).at(beamIdx).back())
        {
            isLastBlockSharedAmongBeams = false;
            break;
        }
    }

    bool beamWidthChanged = (beamWidth != 1) && (isLastBlockSharedAmongBeams != shareAmongBeams);

    // The last block of sequence keeps the memory of recurrent states.
    // When extending the block chain, we insert a placeholder block prior to the last block.
    auto placeholder = getFreeBlock(sequence, executor::KvCacheRetentionConfig::kDefaultRetentionPriority, std::nullopt,
        sequence.getTransferMode(), sequence.getDirectory(), /*wantPlaceholder=*/true);
    TLLM_LOG_DEBUG("%s::allocateBlock - Inserting placeholder block %d before last block for sequence %lu",
        mLogPrefix.c_str(), placeholder->getBlockId(), sequence.getRequestId());
    auto& sequenceBlocks = mAllocatedBlocksPerSeq.at(sequence.getRequestId());
    int numBlocksPerBeam = sequence.getCacheBlockIds(mWindowSize).at(0).size();
    std::vector<KVCacheBlock::IdType> lastBlockIds(beamWidth);
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        lastBlockIds[beamIdx] = sequence.getCacheBlockIds(mWindowSize).at(beamIdx).back();
        if (beamWidthChanged)
        {
            TLLM_CHECK(lastBlockIds[beamIdx] == lastBlockIds[0]);
        }
    }
    // pop last block from all beams
    sequence.removeLastBlock(mWindowSize);
    sequenceBlocks.erase(sequenceBlocks.begin() + (numBlocksPerBeam - 1) * beamWidth, sequenceBlocks.end());
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto lastBlockId = lastBlockIds[beamIdx];
        TLLM_CHECK(lastBlockId >= 0);
        TLLM_LOG_DEBUG("%s::allocateBlock - Swapping placeholder with last block %d for beam %d", mLogPrefix.c_str(),
            lastBlockId, beamIdx);
        auto lastBlock = getBlockById(lastBlockId);
        TLLM_CHECK(lastBlockId == lastBlock->getBlockId());

        // swap block keys between placeholder and lastBlock
        auto tmp = placeholder->getBlockKey();
        placeholder->setBlockKey(lastBlock->getBlockKey(), lastBlock->isFull());
        lastBlock->setBlockKey(tmp, placeholder->isFull());

        // insert placeholder and lastBlock in reverse order
        addBlockToBeam(placeholder, sequence, beamIdx);

        // refresh hash values
        placeholder->setHash();
        lastBlock->setHash();

        // balance ref count
        lastBlock->decRefCount();
    }

    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = (beamWidthChanged && beamIdx > 0) ? getFreeBlock(sequence, sequence.getDecodeRetentionPriority(),
                         sequence.getDecodeDurationMs(), sequence.getTransferMode(), sequence.getDirectory())
                                                       : getBlockById(lastBlockIds[beamIdx]);
        addBlockToBeam(block, sequence, beamIdx);
    }
    return true;
}

void WindowBlockManager::allocateBlock(GenerationRequest& sequence, bool shareAmongBeams)
{
    auto const beamWidth = sequence.getBeamWidth();
    auto const requiredBlocks = shareAmongBeams ? 1 : beamWidth;

    if (LinearAttentionMetadata::hasRecurrentStatesCache(mWindowSize))
    {
        // allocateBlock is called in:
        // 1. decoding phase when block boundary is reached
        // 2. context phase when reuse is disabled
        // In both cases, we don't need to consider about reusing.
        if (tryAllocatePlaceholderForLinearAttention(sequence, shareAmongBeams))
        {
            TLLM_LOG_DEBUG("%s::allocateBlock - Allocated placeholder block for linear attention", mLogPrefix.c_str());
            return;
        }
        TLLM_LOG_DEBUG("%s::allocateBlock - Should allocate new block for linear attention", mLogPrefix.c_str());
    }

    TLLM_CHECK_WITH_INFO(hasFreeBlocks(requiredBlocks),
        "Can't allocate new blocks for window size %d. No free blocks left.", mWindowSize);

    if (shareAmongBeams)
    {
        // add same block to all beams
        auto block = getFreeBlock(sequence, sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs(),
            sequence.getTransferMode(), sequence.getDirectory());
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
    else
    {
        // add different block to each beam
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            auto block = getFreeBlock(sequence, sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs(),
                sequence.getTransferMode(), sequence.getDirectory());
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
}

void WindowBlockManager::copyLinearAttentionBlock(GenerationRequest& sequence, LlmRequest const& request)
{
    if (!isRecurrentState())
    {
        return;
    }

    auto const requestId = request.mRequestId;

    // Check if this sequence exists
    if (mAllocatedBlocksPerSeq.find(requestId) == mAllocatedBlocksPerSeq.end())
    {
        TLLM_LOG_WARNING("%s::copyLinearAttentionBlock - Request %lu not found", mLogPrefix.c_str(), requestId);
        return;
    }

    // It points to the next token to be processed/generated
    auto currentPosition
        = request.isContextFinished() ? (request.getNumTokens(0)) : request.getContextCurrentPosition();
    TLLM_LOG_DEBUG("%s::copyLinearAttentionBlock - Request %lu, currentPosition %d", mLogPrefix.c_str(), requestId,
        currentPosition);

    // edge case: promptLen % tokensPerBlock == 0, and this is the first token of decoding phase
    if (currentPosition == request.getPromptLen() + 1 && request.getPromptLen() % mTokensPerBlock == 0)
    {
        if (TLLM_LIKELY(sequence.getBeamWidth() == 1))
        {
            // the block of beam0 is inherited from context phase, no need to copy
            return;
        }
        // copy beam 0 to other beams
        auto beam0Block = getBlockById(sequence.getCacheBlockIds(mWindowSize).at(0).back());
        for (auto beamIdx = 1; beamIdx < sequence.getBeamWidth(); ++beamIdx)
        {
            auto beamBlockId = sequence.getCacheBlockIds(mWindowSize).at(beamIdx).back();
            auto beamBlock = getBlockById(beamBlockId);
            TLLM_LOG_DEBUG("%s::copyLinearAttentionBlock - Onboarding request %lu, block %d to %d", mLogPrefix.c_str(),
                requestId, beam0Block->getBlockId(), beamBlock->getBlockId());
            mTransferManager->onboard(beam0Block, beamBlock, mPools,
                mTokensPerBlock, // Size of each current state block is fixed. Passing TokensPerBlock to tell the
                                 // transfer manager to copy the entire block.
                sequence.getTransferMode(), sequence.getDirectory());
        }
        return;
    }

    // copy only happens in context phase or the corner case above
    if (currentPosition % mTokensPerBlock != 0 || currentPosition > request.getPromptLen() || currentPosition == 0)
    {
        return;
    }

    auto prevBlockIndex = currentPosition / mTokensPerBlock - 1; // signed
    std::set<std::pair<KVCacheBlock::IdType, KVCacheBlock::IdType>> onboardedBlocks;
    for (auto beamIdx = 0; beamIdx < sequence.getBeamWidth(); ++beamIdx)
    {
        auto const& beamBlockIds = sequence.getCacheBlockIds(mWindowSize).at(beamIdx);
        auto prevBlockId = beamBlockIds.at(prevBlockIndex);
        auto prevBlock = getBlockById(prevBlockId);
        if (prevBlock->isPlaceholder())
        {
            TLLM_LOG_DEBUG(
                "%s::copyLinearAttentionBlock - Previous block %d is a placeholder, skip. This usually happens when "
                "chunked context is enabled but reusing is disabled.",
                mLogPrefix.c_str(), prevBlockId);
            continue;
        }
        auto nextBlockIndex = prevBlockIndex + 1;
        KVCacheBlock::IdType nextBlockId = -1;
        BlockPtr nextBlock = nullptr;
        while (nextBlockIndex < static_cast<int>(beamBlockIds.size()))
        {
            nextBlockId = beamBlockIds.at(nextBlockIndex);
            nextBlock = getBlockById(nextBlockId);
            if (nextBlock != nullptr && !nextBlock->isPlaceholder())
            {
                break;
            }
            nextBlockIndex++;
        }
        TLLM_CHECK(nextBlockId != -1);
        if (onboardedBlocks.find({prevBlockId, nextBlockId}) != onboardedBlocks.end())
        {
            continue;
        }
        TLLM_LOG_DEBUG("%s::copyLinearAttentionBlock - Onboarding request %lu, block %d to %d", mLogPrefix.c_str(),
            requestId, prevBlock->getBlockId(), nextBlock->getBlockId());
        mTransferManager->onboard(prevBlock, nextBlock, mPools,
            mTokensPerBlock, // Size of each current state block is fixed. Passing TokensPerBlock to tell the transfer
                             // manager to copy the entire block.
            sequence.getTransferMode(), sequence.getDirectory());
        onboardedBlocks.insert({prevBlockId, nextBlockId});
    }
}

std::pair<SizeType32, std::vector<KVCacheBlock::IdType>> WindowBlockManager::storeBlocks(
    std::vector<BlockKey> blockKeys, std::vector<BlockPtr> const& blocks, bool pinBlocks)
{
    SizeType32 numBlocksStoredForReuse = 0;
    std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());

    // Trim to the shorter of the two inputs so the zip below is always in-bounds.
    auto const numBlocks = std::min(blockKeys.size(), blocks.size());
    blockKeys.resize(numBlocks);

    TLLM_LOG_DEBUG("%s::storeBlocks - %zu blockKeys, %zu blocks", mLogPrefix.c_str(), numBlocks, blocks.size());

    if (numBlocks == 0)
    {
        return {0, {}};
    }

    // Insert (or look up) trie nodes for the entire prefix chain in one pass.
    // This separates structural trie insertion from block-value assignment and
    // allows us to skip occupied slots and continue storing later blocks
    // (rather than stopping on the first collision).
    auto nodeMatches = mLookupTree->insertNodes(blockKeys);

    std::vector<BlockPtr> storedBlocks;
    std::vector<KVCacheBlock::IdType> pinnedBlockIds;
    // prevBlock tracks the trie-level parent used for hash chaining and setPrevBlockInSeq.
    BlockPtr prevBlock = mCachedBlocksRoot;

    for (std::size_t i = 0; i < nodeMatches.exactMatches.size() && i < numBlocks; ++i)
    {
        auto const& node = nodeMatches.exactMatches[i].node;
        auto const& block = blocks[i];
        auto const& blockKey = blockKeys[i];

        if (block->isPlaceholder())
        {
            // Two placeholder flavors coexist at this call site:
            //   1) SWA on-demand placeholders (blockId == kPlaceholderBlockId): the real
            //      OOW block was stored earlier (storeContextBlocks / storeNewBlock) or
            //      has been evicted. Advance prevBlock via the existing trie value if
            //      present; if absent (evicted anchor), continue past it. SWA lookup can
            //      traverse value-less exact nodes once the missing anchor is OOW.
            //   2) Linear-attention placeholders (blockId is a negative per-slot ID from
            //      mAllPlaceholderBlocksById, not kPlaceholderBlockId): these represent
            //      gaps in the recurrent-state chain and must be stored at their trie
            //      slots so that subsequent lookups can traverse through them.
            auto const existing = node->getValue(mWindowSize);
            if (block->getBlockId() == KVCacheBlock::kPlaceholderBlockId)
            {
                // SWA on-demand placeholder path.
                if (existing.has_value() && *existing)
                {
                    TLLM_LOG_DEBUG("%s::storeBlocks - OOW placeholder at %zu, found anchor block %d in trie",
                        mLogPrefix.c_str(), i, (*existing)->getBlockId());
                    prevBlock = *existing;
                    continue;
                }
                TLLM_LOG_DEBUG(
                    "%s::storeBlocks - OOW placeholder at %zu, anchor block evicted; continuing past missing anchor",
                    mLogPrefix.c_str(), i);
                // Keep the hash/back-pointer chain tied to the nearest populated
                // ancestor while leaving the value-less trie node in place for SWA
                // traversal.
                auto walker = node->getParentNode();
                while (walker)
                {
                    auto ancestorValue = walker->getValue(mWindowSize);
                    if (ancestorValue.has_value() && *ancestorValue)
                    {
                        prevBlock = *ancestorValue;
                        break;
                    }
                    walker = walker->getParentNode();
                }
                continue;
            }

            // Linear-attention placeholder path: store at this slot if empty, advance
            // prevBlock regardless.
            if (!existing.has_value())
            {
                TLLM_LOG_DEBUG("%s::storeBlocks - linear-attention placeholder %d: inserting at trie slot",
                    mLogPrefix.c_str(), block->getBlockId());
                block->detachFromLookupNode();
                block->setBlockKey(blockKey, static_cast<SizeType32>(blockKey.uniqueTokens.size()) == mTokensPerBlock);
                block->setPrevBlockInSeq(prevBlock);
                block->attachToLookupNode(node, mWindowSize);
                auto const newHash = BlockKeyHasher()(blockKey, prevBlock->getHash());
                if (block->getHash() != newHash)
                {
                    block->setHash(newHash);
                }
                storedBlocks.push_back(block);
                prevBlock = block;
                ++numBlocksStoredForReuse;
            }
            else
            {
                prevBlock = *existing;
            }
            continue;
        }

        auto const bid = block->getBlockId();
        auto const existing = node->getValue(mWindowSize);

        if (existing.has_value())
        {
            // Trie slot already occupied (block previously stored by this or another sequence).
            // Advance prevBlock and continue. Subsequent blocks may still need
            // storing as children of this node.
            TLLM_LOG_DEBUG("%s::storeBlocks - Block %d: slot occupied by %d, skipping", mLogPrefix.c_str(), bid,
                (*existing)->getBlockId());
            prevBlock = *existing;
        }
        else
        {
            // Empty trie slot — store this block.
            TLLM_LOG_DEBUG("%s::storeBlocks - Block %d: no existing entry, inserting into search structure",
                mLogPrefix.c_str(), bid);

            block->detachFromLookupNode();
            block->setBlockKey(blockKey, static_cast<SizeType32>(blockKey.uniqueTokens.size()) == mTokensPerBlock);
            block->setPrevBlockInSeq(prevBlock);
            block->attachToLookupNode(node, mWindowSize);

            auto const newHash = BlockKeyHasher()(blockKey, prevBlock->getHash());
            if (block->getHash() != newHash)
            {
                TLLM_LOG_DEBUG("#%d block hash %zx -> %zx", bid, block->getHash(), newHash);
                block->setHash(newHash);
            }

            storedBlocks.push_back(block);
            prevBlock = block;
            numBlocksStoredForReuse++;
        }

        if (pinBlocks)
        {
            // If the block has no refs it sits in the eviction policy's free
            // queue. Claim it first so that the later unpinBlocksById /
            // releaseBlock cycle does not create a duplicate queue entry.
            // Pass the block's existing priority and duration so that
            // claimBlock does not clear its retention/expiry metadata.
            if (!prevBlock->hasRefs())
            {
                mEvictionPolicy->claimBlock(prevBlock, prevBlock->getPriority(), prevBlock->getDurationMs());
            }
            prevBlock->incRefCount();
            pinnedBlockIds.push_back(prevBlock->getBlockId());
        }
    }

    if (mEventManager)
    {
        // Linear-attention placeholders can legitimately land in storedBlocks via the
        // empty-slot branch above (they represent gaps in the recurrent-state chain),
        // but KVCacheEventManager::enqueueStoredEvent calls isPrimary() on every entry
        // and isPrimary() asserts on placeholders.  Filter them out before enqueuing.
        std::vector<BlockPtr> nonPlaceholderStoredBlocks;
        nonPlaceholderStoredBlocks.reserve(storedBlocks.size());
        for (auto const& b : storedBlocks)
        {
            if (!b->isPlaceholder())
            {
                nonPlaceholderStoredBlocks.push_back(b);
            }
        }
        if (!nonPlaceholderStoredBlocks.empty())
        {
            mEventManager->enqueueStoredEvent(nonPlaceholderStoredBlocks, mWindowSize);
        }
    }
    return {numBlocksStoredForReuse, pinnedBlockIds};
}

void BlockManager::replaceSharedBlock(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx)
{
    mWindowBlockManagers.at(windowSize).replaceSharedBlock(sequence, blockIdx);
}

void WindowBlockManager::replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx)
{
    auto const requestId = sequence.getRequestId();
    auto const beamWidth = sequence.getBeamWidth();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);

    if (!allocatedBlocks.at((blockIdx + 1) * beamWidth - 1)->isShared())
    {
        return;
    }
    BlockKey blockKey = allocatedBlocks.at(blockIdx * beamWidth)->getBlockKey();
    bool isFull = allocatedBlocks.at(blockIdx * beamWidth)->isFull();

    // Free shared block
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = allocatedBlocks.at(blockIdx * beamWidth + beamIdx);
        block->decRefCount();
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
        }
    }

    // Allocate new blocks
    TLLM_CHECK_WITH_INFO(hasFreeBlocks(beamWidth), "Can't allocate new blocks. No free blocks left.");
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = getFreeBlock(sequence, executor::KvCacheRetentionConfig::kDefaultRetentionPriority, std::nullopt,
            sequence.getTransferMode(), sequence.getDirectory());
        block->incRefCount();
        if (sequence.getCacheBlockIds(mWindowSize).at(beamIdx).size() == 0)
        {
            block->setPrevBlockInSeq(nullptr);
        }
        else
        {
            block->setPrevBlockInSeq(getBlockById(sequence.getCacheBlockIds(mWindowSize)[beamIdx].back()));
        }
        block->setBlockKey(blockKey, isFull);
        block->setHash();
        sequence.changeCacheBlock(mWindowSize, beamIdx, blockIdx, block->getBlockId());
        allocatedBlocks.at(blockIdx * beamWidth + beamIdx) = block;
    }
}

void BlockManager::releaseLastBlock(GenerationRequest& sequence, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).releaseLastBlock(sequence);
}

void WindowBlockManager::releaseLastBlock(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    auto it = allocatedBlocks.rbegin();
    auto& block = *it;
    // Decrease ref count
    block->decRefCount();
    // If ref count is zero, move block to free blocks
    if (!block->hasRefs())
    {
        mEvictionPolicy->releaseBlock(block, true);
    }
    // Remove block from allocated blocks
    allocatedBlocks.pop_back();
    // Remove stored block ids in sequence
    sequence.removeLastBlock(mWindowSize);
}

[[nodiscard]] SizeType32 WindowBlockManager::getNumFreeBlocks() const
{
    auto const numFree = mEvictionPolicy->getNumFreeBlocks(kPrimaryLevel);
    TLLM_CHECK_WITH_INFO(numFree <= getMaxNumBlocks(),
        "%s::getNumFreeBlocks - primary free block count (%d) exceeds total block count (%d). "
        "This indicates a block accounting inconsistency, likely caused by block pool swaps "
        "during offload/onboard operations. numPrimaryBlocks=%d, numSecondaryBlocks=%d",
        mLogPrefix.c_str(), numFree, getMaxNumBlocks(), mNumPrimaryBlocks, mNumSecondaryBlocks);
    return numFree;
}

[[nodiscard]] SizeType32 WindowBlockManager::getNumFreeSecondaryBlocks() const noexcept
{
    return mEvictionPolicy->getNumFreeBlocks(kSecondaryLevel);
}

KvCacheIterationStats WindowBlockManager::getAndResetIterationStats()
{
    std::lock_guard<std::recursive_mutex> lock(mLookupTree->getMutex());
    KvCacheIterationStats stats;

    // Instantaneous gauges
    stats.primaryMaxNumBlocks = getNumPrimaryBlocks();
    stats.primaryFreeNumBlocks = getNumFreeBlocks();
    stats.primaryUsedNumBlocks = stats.primaryMaxNumBlocks - stats.primaryFreeNumBlocks;
    stats.secondaryMaxNumBlocks = getNumSecondaryBlocks();
    stats.secondaryFreeNumBlocks = getNumFreeSecondaryBlocks();
    stats.secondaryUsedNumBlocks = stats.secondaryMaxNumBlocks - stats.secondaryFreeNumBlocks;

    // Compute deltas since last call — context phase
    stats.iterAllocTotalBlocks = mAllocTotalBlocks - mPrevAllocTotalBlocks;
    stats.iterAllocNewBlocks = mAllocNewBlocks - mPrevAllocNewBlocks;
    stats.iterReusedBlocks = mReusedBlocks - mPrevReusedBlocks;
    stats.iterFullReusedBlocks = mFullReusedBlocks - mPrevFullReusedBlocks;
    stats.iterPartialReusedBlocks = mPartialReusedBlocks - mPrevPartialReusedBlocks;
    stats.iterMissedBlocks = mMissedBlocks - mPrevMissedBlocks;

    auto const iterTotal = stats.iterReusedBlocks + stats.iterMissedBlocks;
    stats.iterCacheHitRate
        = iterTotal == 0 ? 0.0f : static_cast<float>(stats.iterReusedBlocks) / static_cast<float>(iterTotal);

    // Generation phase
    stats.iterGenAllocBlocks = mGenAllocBlocks - mPrevGenAllocBlocks;

    // Snapshot current values for next delta
    mPrevAllocTotalBlocks = mAllocTotalBlocks;
    mPrevAllocNewBlocks = mAllocNewBlocks;
    mPrevReusedBlocks = mReusedBlocks;
    mPrevFullReusedBlocks = mFullReusedBlocks;
    mPrevPartialReusedBlocks = mPartialReusedBlocks;
    mPrevMissedBlocks = mMissedBlocks;
    mPrevGenAllocBlocks = mGenAllocBlocks;

    // Transfer stats (collected from transfer manager)
    if (mTransferManager)
    {
        auto transferStats = mTransferManager->getAndResetTransferStats();
        stats.iterOnboardBlocks = transferStats.onboardBlocks;
        stats.iterOnboardBytes = transferStats.onboardBytes;
        stats.iterOffloadBlocks = transferStats.offloadBlocks;
        stats.iterOffloadBytes = transferStats.offloadBytes;
        stats.iterIntraDeviceCopyBlocks = transferStats.intraDeviceCopyBlocks;
        stats.iterIntraDeviceCopyBytes = transferStats.intraDeviceCopyBytes;
    }

    return stats;
}

std::map<SizeType32, KvCacheIterationStats> BlockManager::getAndResetIterationStats()
{
    std::map<SizeType32, KvCacheIterationStats> perWindowStats;
    for (auto& [windowSize, manager] : mWindowBlockManagers)
    {
        perWindowStats[windowSize] = manager.getAndResetIterationStats();
    }
    return perWindowStats;
}

std::deque<tle::KVCacheEvent> BlockManager::getLatestEvents(std::optional<std::chrono::milliseconds> timeout) const
{
    return mEventManager ? mEventManager->getEvents(timeout) : std::deque<tle::KVCacheEvent>{};
}

std::vector<KVCacheBlock::IdType> BlockManager::storeBlocksForReuse(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    std::vector<KVCacheBlock::IdType> pinnedBlockIds;
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        pinnedBlockIds = manager.storeBlocksForReuse(sequence, llmRequest, pinBlocks);
    }
    return pinnedBlockIds;
}

std::optional<KVCacheBlock::IdType> BlockManager::releaseBlocks(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    // Released block will be stored when reuse is enabled.
    // Reuse is implied to be enabled if llmRequest is provided.
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;

    for (auto& [_, manager] : mWindowBlockManagers)
    {
        if (!llmRequest.has_value() || llmRequest->isDummyRequest() || sequence.getBeamWidth() > 1
            || mLinearAttentionMetadata.has_value()
            /* Hybrid model we only store context blocks for reuse*/)
        {
            lastStoredId = manager.releaseBlocks(sequence, std::nullopt);
        }
        else
        {
            lastStoredId = manager.releaseBlocks(sequence, llmRequest);
        }
    }
    return lastStoredId;
}

void BlockManager::pinBlocks(GenerationRequest& sequence)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.pinBlocks(sequence);
    }
}

void BlockManager::unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds)
{
    // Use the first window size
    if (mWindowBlockManagers.empty())
    {
        return;
    }
    auto& firstManager = mWindowBlockManagers.begin()->second;
    firstManager.unpinBlocksById(blockIds);
}

void WindowBlockManager::pinBlocks(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    for (auto& block : allocatedBlocks)
    {
        block->incRefCount();
    }
}

void WindowBlockManager::unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds)
{
    if (blockIds.empty())
    {
        return;
    }

    for (auto const& blockId : blockIds)
    {
        TLLM_CHECK_WITH_INFO(blockId >= 0 && static_cast<size_t>(blockId) < mAllBlocksById.size(),
            "Block id %d is out of range", blockId);
        auto block = mAllBlocksById[blockId];
        if (block && block->getBlockId() != KVCacheBlock::kCachedBlocksRootId)
        {
            block->decRefCount();
            if (!block->hasRefs())
            {
                mEvictionPolicy->releaseBlock(block);
            }
        }
    }
}

// Only in TRT path
void BlockManager::storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.storeNewBlock(sequence, llmRequest);
    }
}

void WindowBlockManager::storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    auto constexpr beamIdx = 0;
    auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);

    if (uniqueTokens.size() == 0)
    {
        return;
    }

    // TODO: get the caller to mark tokens as filled / not filled, so that the kv-cache manager doesn't
    // have to guess. Only (length - 1) tokens of the sequence have their kv-state recorded in kv-cache. We assume
    // the last token's state is not filled yet.
    auto const usableSize = static_cast<runtime::SizeType32>(uniqueTokens.size()) - 1;
    if (usableSize % mTokensPerBlock != 0)
    {
        return;
    }
    auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableSize, mTokensPerBlock, true);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);

    // Build beam-0 block pointer vector from mAllocatedBlocksPerSeq.  OOW positions
    // contain placeholders (isPlaceholder()==true); storeBlocks handles them.
    auto const requestId = sequence.getRequestId();
    auto& seqBlocks = mAllocatedBlocksPerSeq.at(requestId);
    auto const beamWidth = sequence.getBeamWidth();
    std::vector<BlockPtr> beam0Blocks;
    beam0Blocks.reserve(seqBlocks.size() / std::max(beamWidth, 1));
    for (SizeType32 bi = 0; bi < static_cast<SizeType32>(seqBlocks.size()); bi += beamWidth)
    {
        beam0Blocks.push_back(seqBlocks[bi]);
    }

    if (blockKeys.size() < 2 || beam0Blocks.size() < blockKeys.size())
    {
        // store all blocks
        TLLM_LOG_DEBUG("%s::storeNewBlock - store all blocks", mLogPrefix.c_str());
        (void) storeBlocks(std::move(blockKeys), beam0Blocks);
        return;
    }

    auto const& lastBlock = beam0Blocks.at(blockKeys.size() - 1);
    auto const& prevBlock = beam0Blocks.at(blockKeys.size() - 2);

    // If the previous block is a placeholder or not in the radix tree, store all blocks.
    if (prevBlock->isPlaceholder() || prevBlock->getPrevBlock() == nullptr)
    {
        TLLM_LOG_DEBUG("%s::storeNewBlock - store all blocks", mLogPrefix.c_str());
        (void) storeBlocks(std::move(blockKeys), beam0Blocks);
        return;
    }

    if (!lastBlock->isPlaceholder() && lastBlock->getPrevBlock() != nullptr)
    {
        TLLM_LOG_DEBUG("%s::storeNewBlock - no need to store", mLogPrefix.c_str());
        return;
    }
    TLLM_LOG_DEBUG("%s::storeNewBlock - store the last block", mLogPrefix.c_str());
    (void) storeBlocks(std::move(blockKeys), beam0Blocks);
}

std::vector<KVCacheBlock::IdType> WindowBlockManager::storeBlocksForReuse(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    auto constexpr beamIdx = 0;
    auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);

    // Respect chunked prefill: getUsableUniqueTokenCountForReuse caps at
    // contextCurrentPosition when prefill is not yet complete.  storeBlocks already
    // handles trie slots that are already occupied (skips re-insertion and still pins
    // existing blocks when pinBlocks=true), so callers can invoke storeBlocksForReuse
    // regardless of whether earlier blocks are already in the trie.  See
    // storeContextBlocks for the legacy-test fallback used when contextCurrentPosition==0.
    auto usableUniqueTokenCount = getUsableUniqueTokenCountForReuse(uniqueTokens, *llmRequest);
    if (usableUniqueTokenCount == 0 && llmRequest->getContextCurrentPosition() == 0 && !uniqueTokens.empty())
    {
        usableUniqueTokenCount = static_cast<SizeType32>(uniqueTokens.size()) - 1;
    }
    if (isRecurrentState())
    {
        usableUniqueTokenCount = std::min(
            llmRequest->getPromptLen() - 1, usableUniqueTokenCount); // TODO: enable store for completed sequences
    }
    TLLM_LOG_DEBUG("%s::storeBlocksForReuse: req=%lu, windowSize=%d, uniqueTokens.size()=%zu, usableSize=%zu",
        mLogPrefix.c_str(), llmRequest->mRequestId, mWindowSize, uniqueTokens.size(), usableUniqueTokenCount);
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableUniqueTokenCount, mTokensPerBlock, true);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);

    // Build beam-0 block pointer vector from mAllocatedBlocksPerSeq.  OOW positions are
    // placeholders; storeBlocks handles them transparently.
    auto& seqBlocks = mAllocatedBlocksPerSeq.at(sequence.getRequestId());
    auto const beamWidth = sequence.getBeamWidth();
    std::vector<BlockPtr> beam0Blocks;
    beam0Blocks.reserve(seqBlocks.size() / std::max(beamWidth, 1));
    for (SizeType32 bi = 0; bi < static_cast<SizeType32>(seqBlocks.size()); bi += beamWidth)
    {
        beam0Blocks.push_back(seqBlocks[bi]);
    }

    auto [numStored, pinnedBlockIds] = storeBlocks(std::move(blockKeys), beam0Blocks, pinBlocks);

    return pinnedBlockIds;
}

std::optional<KVCacheBlock::IdType> WindowBlockManager::releaseBlocks(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    auto const requestId = sequence.getRequestId();
    TLLM_LOG_DEBUG("%s::releaseBlocks - requestId=%lu, llmRequest.id=%s", mLogPrefix.c_str(), requestId,
        llmRequest.has_value() ? std::to_string(llmRequest->mRequestId).c_str() : "null");
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    auto node = mAllocatedBlocksPerSeq.extract(requestId);
    TLLM_CHECK(node);
    auto& allocatedBlocks = node.mapped();
    if (llmRequest.has_value() && !isRecurrentState()) // only store context blocks for recurrent states
    {
        // If llmRequest is provided, block store for reuse is enabled. OOW positions in
        // allocatedBlocks are placeholders; storeBlocks handles them (advances past
        // placeholders in the trie rather than re-inserting).
        if (mIsSWA)
        {
            TLLM_LOG_DEBUG("%s::releaseBlocks - SWA sequence %lu, storing blocks for reuse", mLogPrefix.c_str(),
                sequence.getRequestId());
        }
        auto const& uniqueTokens = llmRequest->getUniqueTokens(/*beamIdx=*/0);
        // Respect chunked prefill (see storeContextBlocks for detail + legacy-test fallback).
        SizeType32 usableUniqueTokenCount = getUsableUniqueTokenCountForReuse(uniqueTokens, *llmRequest);
        if (usableUniqueTokenCount == 0 && llmRequest->getContextCurrentPosition() == 0 && !uniqueTokens.empty())
        {
            usableUniqueTokenCount = static_cast<SizeType32>(uniqueTokens.size()) - 1;
        }
        auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(
            uniqueTokens, usableUniqueTokenCount, mTokensPerBlock, /*allowPartial=*/true);
        auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);

        // Build beam-0 block pointer vector directly from allocatedBlocks.
        auto const beamWidth = sequence.getBeamWidth();
        std::vector<BlockPtr> beam0Blocks;
        beam0Blocks.reserve(allocatedBlocks.size() / std::max(beamWidth, 1));
        for (SizeType32 bi = 0; bi < static_cast<SizeType32>(allocatedBlocks.size()); bi += beamWidth)
        {
            beam0Blocks.push_back(allocatedBlocks[bi]);
        }

        auto [numBlocksStoredForReuse, pinnedBlockIds]
            = storeBlocks(std::move(blockKeys), beam0Blocks, /*pinBlocks=*/false);
        TLLM_LOG_DEBUG("%s::releaseBlocks Request %lu, %d blocks stored for reuse", mLogPrefix.c_str(),
            sequence.getRequestId(), numBlocksStoredForReuse);
    }
    // Iterate all allocated blocks (including placeholder sentinels at OOW positions);
    // EvictionPolicy::releaseBlock silently skips placeholders.
    for (auto it = allocatedBlocks.rbegin(); it != allocatedBlocks.rend(); ++it)
    {
        auto& block = *it;
        // Decrease ref count
        if (block->hasRefs())
        {
            // An out-of-window block may not have any ref count.
            block->decRefCount();
        }
        // If ref count is zero, move block to free blocks. Placeholder blocks have
        // mRefCount==0 and are silently ignored by EvictionPolicy::releaseBlock().
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
        }
    }
    // Remove stored block ids in sequence
    sequence.clearCacheBlocks(mWindowSize);
    return lastStoredId;
}

void BlockManager::schedulingReleaseBlocks(RequestIdType requestId)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.schedulingReleaseBlocks(requestId);
    }
}

void WindowBlockManager::schedulingReleaseBlocks(RequestIdType requestId)
{
    for (auto& block : mAllocatedBlocksPerSeq.at(requestId))
    {
        // Skip placeholder blocks: they are OOW sentinels whose mSchedulingRefCount is
        // always 0. Calling decSchedulingRefCount() on them would underflow.
        if (block->isPlaceholder())
        {
            continue;
        }
        // Decrease ref count
        block->decSchedulingRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasSchedulingRefs())
        {
            mSchedulingNumFreeBlocks++;
        }
    }
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, int64_t stream, runtime::SizeType32 maxSequenceLength, SizeType32 chunkSize,
    bool enableBlockReuse, CacheType cacheType, bool enablePartialReuse, bool copyOnPartialReuse,
    bool enableIndexerKCache, SizeType32 indexerKCacheQuantBlockSize, SizeType32 indexerKCacheIndexHeadDim,
    bool indexerKCacheUseFp4, std::optional<LinearAttentionMetadata> linearAttentionMetadata)
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, dtype, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength, chunkSize,
        enableBlockReuse, cacheType, std::nullopt, nullptr, enablePartialReuse, copyOnPartialReuse, nullptr,
        enableIndexerKCache, indexerKCacheQuantBlockSize, indexerKCacheIndexHeadDim, indexerKCacheUseFp4,
        linearAttentionMetadata)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, int64_t stream, runtime::SizeType32 maxSequenceLength, SizeType32 chunkSize,
    bool enableBlockReuse, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager, bool enableIndexerKCache,
    SizeType32 indexerKCacheQuantBlockSize, SizeType32 indexerKCacheIndexHeadDim, bool indexerKCacheUseFp4,
    std::optional<LinearAttentionMetadata> linearAttentionMetadata)
    : KVCacheManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
        maxAttentionWindowVec, dtype, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength, chunkSize,
        enableBlockReuse, cacheType, secondaryOffloadMinPriority, eventManager, enablePartialReuse, copyOnPartialReuse,
        kvCacheConnectorManager, enableIndexerKCache, indexerKCacheQuantBlockSize, indexerKCacheIndexHeadDim,
        indexerKCacheUseFp4, linearAttentionMetadata)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, CudaStreamPtr stream, runtime::SizeType32 maxSequenceLength, SizeType32 chunkSize,
    bool enableBlockReuse, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager, bool enableIndexerKCache,
    SizeType32 indexerKCacheQuantBlockSize, SizeType32 indexerKCacheIndexHeadDim, bool indexerKCacheUseFp4,
    std::optional<LinearAttentionMetadata> linearAttentionMetadata)
    : mMaxBeamWidth(maxBeamWidth)
    , mDataType(dtype)
    , mMaxAttentionWindow(*std::max_element(maxAttentionWindowVec.begin(), maxAttentionWindowVec.end()))
    , mTokensPerBlock(tokensPerBlock)
    , mSinkBubbleLength(BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock))
    , mSinkBlockTokenLength(mSinkBubbleLength + sinkTokenLength)
    , mChunkSize(chunkSize)
    , mBlockManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
          std::move(stream), maxSequenceLength, maxBeamWidth, maxAttentionWindowVec, dtype, mSinkBubbleLength,
          mChunkSize, cacheType, secondaryOffloadMinPriority, std::move(eventManager), enablePartialReuse,
          copyOnPartialReuse, std::move(kvCacheConnectorManager), std::nullopt, enableIndexerKCache,
          indexerKCacheQuantBlockSize, indexerKCacheIndexHeadDim, indexerKCacheUseFp4, linearAttentionMetadata)
    // disable block reuse for sink bubble since chopVectorIntoBlocks does not match KV cache blocks in this case
    , mEnableBlockReuse{mSinkBubbleLength > 0 ? false : enableBlockReuse}
{
    // When num_layers < len(maxAttentionWindowVec), not all window sizes in the
    // repeating pattern are used. Update mMaxAttentionWindow to the actual
    // maximum window size that has been allocated in the block manager.
    mMaxAttentionWindow = mBlockManager.getLastWindowSize();

    TLLM_CHECK_WITH_INFO(mSinkBlockTokenLength == 0 && mSinkBubbleLength == 0,
        "[kv cache manager] streamLLM is not supported at the moment");
    // The sink tokens are stored in blocks separate from other tokens.
    // If the last block of sink tokens is only partially filled,
    // we fill that block with a "bubble" to reach the number of tokens per block.

    TLLM_CHECK(mSinkBlockTokenLength % tokensPerBlock == 0);
    TLLM_LOG_DEBUG("KV cache block reuse is %s", mEnableBlockReuse ? "enabled" : "disabled");
    mSequences.reserve(maxNumSequences);
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, CudaStreamPtr stream, runtime::SizeType32 maxSequenceLength, SizeType32 chunkSize,
    bool enableBlockReuse, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager, bool enableIndexerKCache,
    SizeType32 indexerKCacheQuantBlockSize, SizeType32 indexerKCacheIndexHeadDim, bool indexerKCacheUseFp4,
    std::optional<LinearAttentionMetadata> linearAttentionMetadata)
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, dtype, sinkTokenLength, std::move(stream),
        maxSequenceLength, chunkSize, enableBlockReuse, cacheType, secondaryOffloadMinPriority, std::move(eventManager),
        enablePartialReuse, copyOnPartialReuse, std::move(kvCacheConnectorManager), enableIndexerKCache,
        indexerKCacheQuantBlockSize, indexerKCacheIndexHeadDim, indexerKCacheUseFp4, linearAttentionMetadata)
{
}

void KVCacheManager::allocatePools(bool useUvm)
{
    mBlockManager.allocatePools(useUvm);
    auto const numPools = mBlockManager.getNumPools();

    uint64_t cacheSizeBytes = 0;
    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const cacheShape = mBlockManager.getPrimaryPool(poolIdx)->getShape();
        auto const cacheVolume = ITensor::volume(cacheShape);
#ifdef ENABLE_FP4
        auto const isFp4 = mDataType == nvinfer1::DataType::kFP4;
#else
        auto const isFp4 = false;
#endif
        if (!isFp4)
        {
            cacheSizeBytes += cacheVolume * BufferDataType(mDataType).getSize();
        }
        else
        {
            cacheSizeBytes += (cacheVolume * 4) / 8;
        }
    }
    // Save the total number of bytes allocated for the KV-cache for KvCacheStats
    mAllocatedBytes = cacheSizeBytes;
    if (tc::Logger::getLogger()->getLevel() <= tc::Logger::INFO)
    {

        TLLM_LOG_INFO("Number of tokens per block: %d.", mBlockManager.getTokensPerBlock());
        auto const maxNumTokens = mBlockManager.getNumPrimaryBlocks() * mBlockManager.getTokensPerBlock();
        TLLM_LOG_INFO("[MemUsageChange] Allocated %0.2f GiB for max tokens in paged KV cache (%d).",
            cacheSizeBytes / static_cast<double>(1 << 30), maxNumTokens);
    }

    auto const numKVPools
        = mBlockManager.getNumPools(/*include_block_scalar_pools=*/false, /*include_indexer_k_cache_pools=*/false);
    auto const numBlockScalePools
        = mBlockManager.getNumPools(/*includeBlockScalePools=*/true, /*includeIndexerKCachePools=*/false) - numKVPools;

    // Code in the attention kernels is cleaner if we can access the KV values and block scales separately.
    mBlockPoolPointers = BufferManager::cpu(ITensor::makeShape({numKVPools, 2}), TRTDataType<void*>::value);
    mBlockScalePoolPointers
        = BufferManager::cpu(ITensor::makeShape({numBlockScalePools, 2}), TRTDataType<void*>::value);

    auto poolPtrsRange = BufferRange<void*>(*mBlockPoolPointers);
    auto blockScalePtrsRange = BufferRange<void*>(*mBlockScalePoolPointers);
    SizeType32 kvPoolIdx = 0;
    SizeType32 blockScalePoolIdx = 0;

    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const& pool = mBlockManager.getPool(poolIdx);
        auto& outIdx = pool.containsBlockScales ? blockScalePoolIdx : kvPoolIdx;
        auto& outRange = pool.containsBlockScales ? blockScalePtrsRange : poolPtrsRange;
        if (pool.containsIndexerKCache)
        {
            mIndexerKCachePoolPointers = pool.primaryPtr;
        }
        else
        {
            outRange[outIdx * 2] = pool.primaryPtr->data();
            outRange[outIdx * 2 + 1] = pool.secondaryPtr ? pool.secondaryPtr->data() : nullptr;
            outIdx++;
        }
    }

    auto const numLayers = mBlockManager.getNumLayers();
    mLayerToPoolMapping = BufferManager::cpu(ITensor::makeShape({numLayers, 2}), TRTDataType<SizeType32>::value);
    auto poolMappingRange = BufferRange<SizeType32>(*mLayerToPoolMapping);
    for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        auto const indexOfPool = mBlockManager.getLayerPoolIdx(layerIdx);
        auto const layerIdxInCachePool = mBlockManager.getPoolLayerIdx(layerIdx);
        poolMappingRange[layerIdx * 2] = indexOfPool;
        poolMappingRange[layerIdx * 2 + 1] = layerIdxInCachePool;
    }
}

void KVCacheManager::releasePools()
{
    mBlockManager.releasePools();
}

void KVCacheManager::startScheduling()
{
    mBlockManager.startScheduling();
}

SizeType32 KVCacheManager::getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead, SizeType32 windowSize,
    std::optional<PrefixReuseSummary> const& cachedSummary) const
{
    // Default to zero; overwritten below when block reuse is active for a first-chunk context request.
    req.setEstimatedReusableTokens(0);

    if ((req.isContextInitState() && req.isFirstContextChunk()) || req.isDisaggGenerationInitState())
    {
        auto const maxDraftTokensToAdd = std::min(req.getNumDraftTokens(), req.mMaxNewTokens);
        auto const promptCacheLen
            = std::min((isCrossKv() ? req.getEncoderOutputLen() : req.mPromptLen) + maxDraftTokensToAdd,
                  windowSize + mChunkSize)
            + mSinkBubbleLength;
        auto const numSharedBlocks = promptCacheLen / getTokensPerBlock();
        auto const numUnSharedTokens = promptCacheLen % getTokensPerBlock();
        auto const numUnSharedBlocks
            = tc::ceilDiv(numUnSharedTokens, getTokensPerBlock()) * req.mSamplingConfig.beamWidth;
        auto numRequiredBlocks = numSharedBlocks + numUnSharedBlocks;

        // Subtract reusable blocks if block reuse is enabled and we're not using variable window attention
        if (mEnableBlockReuse && !mBlockManager.isVariableWindow() && !isCrossKv()
            && !req.isDisaggGenerationInitState())
        {
            // Use the cached summary if provided; otherwise perform a fresh tree walk.
            auto const summary
                = cachedSummary.has_value() ? cachedSummary.value() : analyzePrefixReuse(req.getUniqueTokens(0), req);
            auto const numReusableBlocks = summary.reusableBlocksAllocated;
            auto const promptInputLen = std::min(req.mPromptLen, windowSize + mChunkSize);
            // Sequence insertion ignores the last prompt token because its KV cannot be recovered.
            // When the prompt lands exactly on a block boundary, counting reusable full blocks from
            // all unique tokens can over-credit one extra shared block.
            TLLM_CHECK_WITH_INFO(promptInputLen > 0, "Unexpected: promptInputLen == 0");
            auto const maxRecoverableSharedBlocks = (promptInputLen - 1) / getTokensPerBlock();
            // Only subtract from shared blocks (reusable blocks are always shared)
            auto const reusableSharedBlocks
                = std::min(numReusableBlocks, std::min(numSharedBlocks, maxRecoverableSharedBlocks));
            numRequiredBlocks -= reusableSharedBlocks;
            // Store on request so the micro batch scheduler can use it for token budget
            req.setEstimatedReusableTokens(reusableSharedBlocks * getTokensPerBlock());
        }
        return numRequiredBlocks;
    }

    if (req.isGenerationInProgressState())
    {
        if (isCrossKv())
        {
            return 0;
        }

        auto const numCurrTokens = getSequence(req.mRequestId).getNumTokens();
        auto const generatedTokens = numCurrTokens - req.getPromptLen();
        auto const maxTokensToAddToKVCache = req.mMaxNewTokens - generatedTokens;
        auto const tokensPerStep = req.getNumDraftTokens() + 1;
        auto const maxTokensToAdd = std::min((twoStepsLookAhead ? 2 : 1) * tokensPerStep, maxTokensToAddToKVCache);
        auto const numNextTokens = numCurrTokens + maxTokensToAdd;

        if (numNextTokens > mBlockManager.getWindowSizeMetadata(windowSize).maxTokenNum)
        {
            return 0;
        }

        auto const numCurrBlocks = tc::ceilDiv(numCurrTokens, getTokensPerBlock());
        auto const numNextBlocks = tc::ceilDiv(numNextTokens, getTokensPerBlock());
        auto const numRequiredBlocks = (numNextBlocks - numCurrBlocks) * req.mSamplingConfig.beamWidth;
        return numRequiredBlocks;
    }

    return 0;
}

SizeType32 KVCacheManager::getRemainingBlocksToCompletion(
    LlmRequest const& req, SizeType32 windowSize, std::optional<PrefixReuseSummary> const& cachedSummary) const
{
    // Default to zero; overwritten below when block reuse is active for a first-chunk context request.
    req.setEstimatedReusableTokens(0);

    if (isCrossKv())
    {
        if (req.isContextInitState() && req.getContextCurrentPosition() == 0)
        {
            return tc::ceilDiv(req.getEncoderOutputLen(), getTokensPerBlock());
        }

        return 0; // cross KV cache doesn't grow after the initial context phase
    }

    if (windowSize == LinearAttentionMetadata::kRecurrentStates)
    {
        if (req.isGenerationInProgressState())
        {
            return 0; // no need to allocate blocks for recurrent states during generation
        }
        else if (!req.isContextFinished())
        {
            std::scoped_lock lck(mSequencesMtx);
            auto const seqIt = mSequences.find(req.mRequestId);
            if (seqIt != mSequences.end())
            {
                return 0;
            }
            if (mEnableBlockReuse)
            {
                return req.getPromptLen() / mBlockManager.getLinearAttentionMetadata()->statesSnapshotInterval + 1
                    + (mBlockManager.getLinearAttentionMetadata()->saveLastSnapshot ? 1 : 0);
            }
            return 1;
        }
    }

    TLLM_CHECK_WITH_INFO(mSinkBubbleLength == 0, "Sink bubble length is not supported at the moment");

    SizeType32 const numContextBlocks = std::min(req.mPromptLen, windowSize + mChunkSize) / getTokensPerBlock();

    SizeType32 const numTotalBlocksPerBeam
        = tc::ceilDiv(std::min(req.mPromptLen + req.mMaxNewTokens, windowSize + mChunkSize), getTokensPerBlock());

    SizeType32 const numGenBlocksPerBeam = numTotalBlocksPerBeam - numContextBlocks;

    SizeType32 numAllocBlocksPerBeam = 0;
    {
        std::scoped_lock lck(mSequencesMtx);
        auto const seqIt = mSequences.find(req.mRequestId);
        if (seqIt != mSequences.end())
        {
            auto const& seq = seqIt->second;
            // Subtract detached front blocks (from SWA sliding) which are still
            // in the cache block ID list but no longer held by the sequence.
            numAllocBlocksPerBeam = seq.getCacheBlockIds(windowSize).at(0).size() - seq.getNumFrontBlocksRemoved();
        }
    }

    // Only subtract reusable blocks that are already allocated (have active refs). See the
    // comment in getNeededBlocksOneStep for the full rationale — free reusable blocks must
    // not be subtracted because they are already counted in the eviction policy's free count.
    SizeType32 numReusableContextBlocks = 0;
    if (mEnableBlockReuse && !mBlockManager.isVariableWindow() && req.isContextInitState() && req.isFirstContextChunk()
        && numAllocBlocksPerBeam == 0)
    {
        // Use the cached summary if provided; otherwise perform a fresh tree walk.
        auto const summary
            = cachedSummary.has_value() ? cachedSummary.value() : analyzePrefixReuse(req.getUniqueTokens(0), req);
        // Block budget: only subtract blocks that are already allocated (have active refs).
        // Free cached blocks are already counted in the eviction policy's free pool and
        // must not be double-counted against the capacity estimate.
        // Cap at (promptLen-1)/tpb to avoid over-counting when the prompt is exactly
        // block-aligned (last full block has not been committed to the tree yet).
        SizeType32 const maxRecoverableBlocks = (req.mPromptLen - 1) / getTokensPerBlock();
        numReusableContextBlocks = std::min({summary.reusableBlocksAllocated, numContextBlocks, maxRecoverableBlocks});
        // Token budget: count all reusable blocks (free or allocated). Cached tokens need
        // not be recomputed regardless of whether their blocks currently have active refs.
        req.setEstimatedReusableTokens(
            std::min({summary.reusableBlocksAll, numContextBlocks, maxRecoverableBlocks}) * getTokensPerBlock());
        TLLM_LOG_DEBUG(
            "getRemainingBlocksToCompletion: request ID %lu, numContextBlocks=%d, "
            "numReusableBlocksAllocated=%d, numReusableBlocksAll=%d, "
            "numReusableContextBlocks=%d, numGenBlocksPerBeam=%d",
            req.mRequestId, numContextBlocks, summary.reusableBlocksAllocated, summary.reusableBlocksAll,
            numReusableContextBlocks, numGenBlocksPerBeam);
    }

    // In case of sliding window attention, a new block is allocated when the
    // window slides (and then the out-of-window block is detached). So we
    // need an extra block for generation if the diff between the max sequence
    // length and the current sequence length crosses both a block boundary
    // and a window boundary.
    auto const isSlidingWindow = (req.mPromptLen + req.mMaxNewTokens) > windowSize;
    SizeType32 const currentSeqlenInBlocks = tc::ceilDiv(req.getNumTokens(0), getTokensPerBlock());
    SizeType32 const maxSeqlenInBlocks = tc::ceilDiv(req.mPromptLen + req.mMaxNewTokens, getTokensPerBlock());
    auto const willCrossBlockBoundary = maxSeqlenInBlocks > currentSeqlenInBlocks;
    auto const willCrossWindowBlockBoundary = maxSeqlenInBlocks > numTotalBlocksPerBeam;
    SizeType32 numExtraBlocksPerBeam
        = isSlidingWindow && willCrossBlockBoundary && willCrossWindowBlockBoundary ? 1 : 0;

    // Adjust for reusable context blocks (only allocated ones)
    SizeType32 const effectiveContextBlocks = numContextBlocks - numReusableContextBlocks;

    if (numAllocBlocksPerBeam < effectiveContextBlocks) // Still haven't allocated all context blocks
    {
        return effectiveContextBlocks - numAllocBlocksPerBeam
            + (numGenBlocksPerBeam + numExtraBlocksPerBeam) * req.mSamplingConfig.beamWidth;
    }

    SizeType32 const effectiveTotalBlocks = numTotalBlocksPerBeam - numReusableContextBlocks;
    return (effectiveTotalBlocks - numAllocBlocksPerBeam + numExtraBlocksPerBeam) * req.mSamplingConfig.beamWidth;
}

void BlockManager::updateSequenceCacheBlockOffsets(GenerationRequest& sequence, SizeType32 windowSize)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(windowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(windowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        for (SizeType32 blockIdx = 0; blockIdx < static_cast<SizeType32>(beamCacheBlock.size()); ++blockIdx)
        {
            auto const blockId = beamCacheBlock.at(blockIdx);
            mWindowBlockManagers.at(windowSize).setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
        }
    }
}

void WindowBlockManager::updateLastCacheBlockOffsets(GenerationRequest& sequence)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(mWindowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(mWindowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.back();
        auto const blockIdx = static_cast<SizeType32>(beamCacheBlock.size() - 1);
        setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
    }
}

void BlockManager::updateCacheBlockOffsetsAtIdx(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(windowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(windowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.at(blockIdx);
        mWindowBlockManagers.at(windowSize).setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
    }
}

void KVCacheManager::addToken(RequestIdType requestId)
{
    // TODO: add streamLLM support
    auto& sequence = getSequence(requestId);
    sequence.addNewTokens(1);
    mBlockManager.adjustBlocksIfNeeded(sequence);
}

void KVCacheManager::copyLinearAttentionBlock(LlmRequest const& llmRequest)
{
    auto& sequence = getSequence(llmRequest.mRequestId);
    mBlockManager.copyLinearAttentionBlock(sequence, llmRequest);
}

void WindowBlockManager::detachFrontBlock(GenerationRequest& sequence)
{
    // streamLLM is not supported at the moment. The out of window block will
    // always be the 0th block.
    TLLM_CHECK_WITH_INFO(
        sequence.getBeamWidth() == 1, "[kv cache manager] detachBlock does not support beamWidth > 1 now.");

    auto const requestId = sequence.getRequestId();
    auto const beamWidth = sequence.getBeamWidth();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    SizeType32 outOfWindowBlockIdx = sequence.getNumFrontBlocksRemoved();

    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto& blockSlot = allocatedBlocks.at(outOfWindowBlockIdx * beamWidth + beamIdx);
        auto outOfWindowBlock = blockSlot;
        TLLM_LOG_DEBUG("%s::detachFrontBlock - Detaching block %d from sequence %d", mLogPrefix.c_str(),
            outOfWindowBlock->getBlockId(), requestId);

        // Replace the real block in mAllocatedBlocksPerSeq with an on-demand SWA
        // placeholder so that subsequent storeBlocks / storeNewBlock calls see a
        // placeholder at this OOW position and advance via lookup, rather than
        // trying to re-insert the real block. Use the
        // kPlaceholderBlockId sentinel (not the real block's ID) to avoid
        // mAllBlocksById aliasing.
        blockSlot = KVCacheBlock::createPlaceholder();

        // Clear retention duration and expiration so the block is treated as a plain
        // cache block at its user-assigned priority, not one with an expiry that
        // originated from its prior in-window state (coderabbitai review #2902723423).
        // We intentionally DO NOT override setPriority here: the block's priority is
        // preserved so that subsequent getFreeBlock / eviction decisions honor the
        // priority level the user originally requested for the block (thorjohnsen
        // review #2902723423).
        outOfWindowBlock->setDurationMs(std::nullopt);
        outOfWindowBlock->setExpirationTime(std::nullopt);

        outOfWindowBlock->decRefCount();

        if (outOfWindowBlock->hasRefs())
        {

            TLLM_LOG_DEBUG("%s::detachFrontBlock - OOW Block %d still has a non-zero ref count", mLogPrefix.c_str(),
                outOfWindowBlock->getBlockId());
        }
        if (!outOfWindowBlock->hasRefs())
        {
            mEvictionPolicy->releaseBlock(outOfWindowBlock);
        }
    }

    // Disconnect first block from sequence and remove it from allocated blocks
    sequence.removeFrontBlock(mWindowSize);
}

PrefixReuseSummary KVCacheManager::analyzePrefixReuse(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    return mBlockManager.analyzePrefixReuse(uniqueTokens, llmRequest);
}

void KVCacheManager::addSequenceBatch(
    std::vector<std::tuple<LlmRequest::RequestIdType, SizeType32, SizeType32>> const& requestInfos,
    std::vector<std::reference_wrapper<LlmRequest>> const& llmRequests)
{
    TLLM_CHECK(requestInfos.size() == llmRequests.size());
    if (requestInfos.empty())
    {
        return;
    }
    auto const n = requestInfos.size();

    // --- Setup: create sequences, hold them (window-independent) ---
    std::vector<GenerationRequest*> sequences(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto const requestId = std::get<0>(requestInfos[i]);
        auto const inputLength = std::get<1>(requestInfos[i]);
        auto const beamWidth = std::get<2>(requestInfos[i]);
        auto& llmRequest = llmRequests[i].get();

        auto kvCacheRetentionConfig
            = llmRequest.getKvCacheRetentionConfig().value_or(executor::KvCacheRetentionConfig());

        auto const [seqIt, emplaceDone] = [&]
        {
            auto lck = std::scoped_lock(mSequencesMtx);
            return mSequences.try_emplace(requestId, requestId, inputLength, beamWidth,
                mBlockManager.getWindowSizesMetadata(), kvCacheRetentionConfig);
        }();
        TLLM_CHECK(emplaceDone);

        sequences[i] = &seqIt->second;
    }

    // Track the minimum prepopulated length across all windows per sequence
    // (for VSWA with mixed isSWA flags).
    std::vector<SizeType32> minPrepopulatedLen(n, std::numeric_limits<SizeType32>::max());
    // Accumulate block allocation stats across all windows per sequence
    std::vector<SizeType32> totalAllocTotalDelta(n, 0);
    std::vector<SizeType32> totalAllocNewDelta(n, 0);
    std::vector<SizeType32> totalReusedDelta(n, 0);
    std::vector<SizeType32> totalMissedDelta(n, 0);

    // --- Iterate over all window sizes (single iteration for non-VSWA) ---
    for (auto const& [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        // Use maxTokenNum (= max(windowSize, maxSequenceLength) + sinkBubbleLength) as the cap.
        // For SWA, blocks are allocated linearly for the full prompt; out-of-window blocks are
        // only detached during generation in adjustBlocksIfNeeded.
        auto const maxTokenNum = metadata.maxTokenNum;

        // Compute per-sequence effective input length for this window
        std::vector<SizeType32> inputLengths(n);
        std::vector<SizeType32> numContextBlocksVec(n);
        for (size_t i = 0; i < n; ++i)
        {
            auto const inputLength = std::get<1>(requestInfos[i]);
            inputLengths[i] = std::min(inputLength, maxTokenNum);
            numContextBlocksVec[i] = tc::ceilDiv(inputLengths[i], getTokensPerBlock());
        }

        // Two-phase claim-then-onboard for this window
        auto const windowResults = mBlockManager.addSequenceBatch(
            sequences, inputLengths, numContextBlocksVec, llmRequests, windowSize, mEnableBlockReuse);

        // Update offsets and accumulate stats
        for (size_t i = 0; i < n; ++i)
        {
            mBlockManager.updateSequenceCacheBlockOffsets(*sequences[i], windowSize);

            auto const& stats = windowResults[i];
            minPrepopulatedLen[i] = std::min(minPrepopulatedLen[i], stats.prepopulatedLen);
            totalAllocTotalDelta[i] += stats.allocTotalDelta;
            totalAllocNewDelta[i] += stats.allocNewDelta;
            totalReusedDelta[i] += stats.reusedDelta;
            totalMissedDelta[i] += stats.missedDelta;
        }
    }

    // --- Finalize: set prepopulated length and per-request stats ---
    for (size_t i = 0; i < n; ++i)
    {
        auto& llmRequest = llmRequests[i].get();

        if (mEnableBlockReuse)
        {
            TLLM_LOG_DEBUG("KVCacheManager::addSequenceBatch: Setting prepopulatedPromptLen to %d for request %lu",
                minPrepopulatedLen[i], llmRequest.mRequestId);
            llmRequest.setPrepopulatedPromptLen(minPrepopulatedLen[i], getTokensPerBlock());
            // Clear the scheduling estimate now that the authoritative value is set.
            // This prevents subsequent chunks from double-counting reusable tokens.
            llmRequest.setEstimatedReusableTokens(0);
        }

        llmRequest.updateAllocTotalBlocksPerRequest(totalAllocTotalDelta[i]);
        llmRequest.updateAllocNewBlocksPerRequest(totalAllocNewDelta[i]);
        llmRequest.updateReusedBlocksPerRequest(totalReusedDelta[i]);
        llmRequest.updateMissedBlocksPerRequest(totalMissedDelta[i]);
    }
}

void KVCacheManager::storeContextBlocks(LlmRequest const& llmRequest)
{
    auto const requestId = llmRequest.mRequestId;
    bool found = false;
    {
        // protect the mSequences
        std::scoped_lock lock(mSequencesMtx);
        found = mSequences.find(requestId) != mSequences.end();
    }
    if (found)
    {
        auto& sequence = getSequence(requestId);
        if (mEnableBlockReuse && !llmRequest.isDummyRequest())
        {
            mBlockManager.storeContextBlocks(sequence, llmRequest);
        }
    }
    else
    {
        TLLM_LOG_WARNING("[kv cache manager] storeContextBlocks: Can not find sequence for request %lu", requestId);
    }
}

// Only in TRT path
void KVCacheManager::storeNewBlock(LlmRequest const& llmRequest)
{
    // We store newest block for potential reuse only if:
    // - Beam search is NOT enabled
    // - Block reuse is enabled.
    auto const requestId = llmRequest.mRequestId;
    auto& sequence = getSequence(requestId);
    if (sequence.getBeamWidth() > 1 || !mEnableBlockReuse)
    {
        return;
    }
    mBlockManager.storeNewBlock(sequence, llmRequest);
}

std::optional<KVCacheBlock::IdType> KVCacheManager::removeSequence(
    RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    TLLM_LOG_TRACE("[%s]::%s start", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    auto sequenceNode = [this, requestId]
    {
        std::scoped_lock lock(mSequencesMtx);
        return mSequences.extract(requestId);
    }();
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    if (!sequenceNode.empty())
    {
        if (mEnableBlockReuse)
        {
            lastStoredId = mBlockManager.releaseBlocks(sequenceNode.mapped(), llmRequest, pinBlocks);
        }
        else
        {
            lastStoredId = mBlockManager.releaseBlocks(sequenceNode.mapped(), std::nullopt, pinBlocks);
        }
    }
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    return lastStoredId;
}

std::vector<KVCacheBlock::IdType> KVCacheManager::storeBlocksForReuse(
    RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    TLLM_LOG_TRACE("[%s]::%s start", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    auto& sequence = getSequence(requestId);
    auto pinnedBlockIds = mBlockManager.storeBlocksForReuse(sequence, llmRequest, pinBlocks);
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    return pinnedBlockIds;
}

void KVCacheManager::schedulingRemoveSequence(RequestIdType requestId)
{
    // Mimic Free all blocks for this sequence
    mBlockManager.schedulingReleaseBlocks(requestId);
}

void KVCacheManager::pinBlocks(RequestIdType requestId)
{
    auto& sequence = getSequence(requestId);
    mBlockManager.pinBlocks(sequence);
}

void KVCacheManager::unpinBlocksById(std::vector<KVCacheBlock::IdType> const& blockIds)
{
    mBlockManager.unpinBlocksById(blockIds);
}

tle::RetentionPriority KVCacheManager::getPriorityByBlockId(KVCacheBlock::IdType blockId, SizeType32 windowSize) const
{
    try
    {
        BlockPtr const& block = mBlockManager.getBlockById(blockId, windowSize);
        if (block)
        {
            return block->getPriority();
        }
        TLLM_LOG_WARNING("getPriorityByBlockId: Block ID %d not found in window %d", blockId, windowSize);
        return tle::KvCacheRetentionConfig::kDefaultRetentionPriority;
    }
    catch (std::out_of_range const& ex)
    {
        TLLM_LOG_WARNING(
            "getPriorityByBlockId: Block ID %d or window size %d out of range: %s", blockId, windowSize, ex.what());
        return tle::KvCacheRetentionConfig::kDefaultRetentionPriority;
    }
}

SizeType32 KVCacheManager::copyBlockOffsets(ITensor& output, SizeType32 outputSlotOffset, RequestIdType requestId) const
{
    auto const& sequence = getSequence(requestId);
    auto const beamWidth = sequence.getBeamWidth();

    auto* dstPtr = bufferCast<tk::KVCacheIndex>(output);
    auto const& dstShape = output.getShape();

    SizeType32 constexpr kIdx = 0;
    SizeType32 constexpr vIdx = 1;

    SizeType32 maxBlockCount{0};
    // Get page table for each KV cache pool
    SizeType32 absolutePoolIdx = 0;

    for (auto const [ws, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        auto const& cacheBlocksTensor = sequence.getCacheBlockIndices(ws);
        auto const* srcPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
        auto const& srcShape = cacheBlocksTensor.getShape();
        auto const& cacheBlockIds = sequence.getCacheBlockIds(ws);
        for (SizeType32 poolIdx = 0; poolIdx < metadata.numPools; poolIdx++, absolutePoolIdx++)
        {
            for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const beamBlockCount = cacheBlockIds[beamIdx].size();
                auto const copyChunkSize = beamBlockCount * sizeof(tk::KVCacheIndex);
                for (auto xIdx : {kIdx, vIdx})
                {
                    auto const srcIndex = tc::flat_index(srcShape.d, poolIdx, beamIdx, xIdx, 0);
                    auto const dstIndex
                        = tc::flat_index(dstShape.d, absolutePoolIdx, outputSlotOffset + beamIdx, xIdx, 0);
                    std::memcpy(dstPtr + dstIndex, srcPtr + srcIndex, copyChunkSize);
                }
                maxBlockCount = std::max<SizeType32>(maxBlockCount, static_cast<SizeType32>(beamBlockCount));
            }
        }
    }
    return maxBlockCount;
}

void KVCacheManager::getBlockOffsetsOfBatch(
    ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
{
    // Get page table for each KV cache pool
    for (auto batchSlotIdx = 0; batchSlotIdx < batchSize; ++batchSlotIdx)
    {
        copyBlockOffsets(output, batchSlotIdx * beamWidth, firstBatchSlotIdx + batchSlotIdx);
    }
}

std::map<SizeType32, std::vector<SizeType32>> BaseKVCacheManager::groupLayersByWindowSize(
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 numLayers)
{
    auto const numNonUniqueWindowSizes = static_cast<SizeType32>(maxAttentionWindowVec.size());
    std::map<SizeType32, std::vector<SizeType32>> uniqueWindowSizeToLayers;
    for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        /*
        At this point (Deep in the construction of TrtGptModel), maxAttentionWindowVec isn't "stretched" to the
        length of numLayers yet. So, we need to rotate the window sizes per layer with modulo.
        */
        auto const windowSize = maxAttentionWindowVec.at(layerIdx % numNonUniqueWindowSizes);
        uniqueWindowSizeToLayers[windowSize].push_back(layerIdx);
    }
    return uniqueWindowSizeToLayers;
}

std::tuple<uint64_t, uint64_t> BaseKVCacheManager::calculateFreeMemBytes(
    runtime::BufferManager const& bufferManager, executor::KvCacheConfig const& config)
{
    auto const freeMemFraction
        = config.getFreeGpuMemoryFraction().value_or(executor::KvCacheConfig::kDefaultGpuMemFraction);
    TLLM_CHECK_WITH_INFO(freeMemFraction < 1.0F,
        "Invalid freeMemFraction, freeMemFraction (%f) must be smaller than 1.0f", freeMemFraction);
    if (config.getMaxTokens().has_value())
    {
        if (config.getFreeGpuMemoryFraction().has_value())
        {
            TLLM_LOG_WARNING(
                "Both freeGpuMemoryFraction (aka kv_cache_free_gpu_mem_fraction) "
                "and maxTokens (aka max_tokens_in_paged_kv_cache) "
                "are set (to %f and %ld, respectively). The smaller value will be used.",
                freeMemFraction, (int64_t) config.getMaxTokens().value());
        }
    }

    TLLM_CUDA_CHECK(::cudaDeviceSynchronize());
    auto const [freeMem, totalMem] = tc::getDeviceMemoryInfo(config.getUseUvm());
    auto const finalFreeMem = freeMem + bufferManager.memoryPoolFree();
    TLLM_LOG_INFO("Memory usage when calculating max tokens in paged kv cache: total: %0.2f GiB, available: %0.2f GiB",
        totalMem / static_cast<double>(1 << 30), finalFreeMem / static_cast<double>(1 << 30));
    TLLM_CHECK_WITH_INFO(finalFreeMem <= totalMem, "Free memory cannot exceed total memory");

    auto const freePrimaryMemBytes = static_cast<uint64_t>(finalFreeMem * freeMemFraction);
    auto const freeSecondaryMemBytes = config.getHostCacheSize().value_or(0);

    TLLM_LOG_DEBUG("Calculated free memory: {.freePrimaryMemBytes=%" PRIu64 ", .freeSecondaryMemBytes=%" PRIu64 "}",
        freePrimaryMemBytes, freeSecondaryMemBytes);

    return std::make_tuple(freePrimaryMemBytes, freeSecondaryMemBytes);
}

namespace
{
bool isSortedVectorIdenticalAcrossAllRanks(WorldConfig const& worldConfig, std::vector<SizeType32> const& vector)
{
    auto const numRanks = worldConfig.getSize();
    auto const numElements = static_cast<int>(vector.size());
    int maxNumElements = 0;
    int minNumElements = 0;
    COMM_SESSION.allreduce(&numElements, &maxNumElements, 1, mpi::MpiType::kINT32, mpi::MpiOp::MAX);
    COMM_SESSION.allreduce(&numElements, &minNumElements, 1, mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    if (maxNumElements != minNumElements)
    {
        return false;
    }
    std::vector<SizeType32> allElements(numElements * numRanks);
    COMM_SESSION.allgather(vector.data(), allElements.data(), numElements, mpi::MpiType::kUINT32);

    for (int i = 0; i < numElements; ++i)
    {
        auto const ref = allElements.at(i);
        for (int rank = 1; rank < numRanks; ++rank)
        {
            if (allElements[rank * numElements + i] != ref)
                return false;
        }
    }
    return true;
}

/// Drive a min-allreduce so all ranks agree on the primary/secondary block counts per window size.
/// No-op for single-rank world configs. @p windowSizes must be sorted identically across ranks.
void synchronizeBlockCountsAcrossRanks(WorldConfig const& worldConfig, std::vector<SizeType32> const& windowSizes,
    std::vector<SizeType32>& blocksPrimary, std::vector<SizeType32>& blocksSecondary)
{
    if (worldConfig.getSize() <= 1)
    {
        return;
    }
    TLLM_CHECK(worldConfig.validMpiConfig());
    auto const rank = worldConfig.getRank();
    using tensorrt_llm::common::vec2str;
    TLLM_CHECK_WITH_INFO(isSortedVectorIdenticalAcrossAllRanks(worldConfig, windowSizes),
        "[RANK %d] Asymmetrical pipeline parallelism detected: Ranks either have a different number of window "
        "sizes, or differing values. This is not supported with Variable Sliding Window Attention. Local window "
        "sizes for reference: %s",
        rank, vec2str(windowSizes).c_str());
    TLLM_LOG_DEBUG(
        "[RANK %d] Before mpi::MpiOp::MIN reduction: window sizes %s / primary blocks %s / secondary blocks %s", rank,
        vec2str(windowSizes).c_str(), vec2str(blocksPrimary).c_str(), vec2str(blocksSecondary).c_str());
    // make sure all ranks use same value for max blocks
    auto blocksWorld = blocksPrimary;
    COMM_SESSION.allreduce(
        blocksPrimary.data(), blocksWorld.data(), windowSizes.size(), mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    blocksPrimary = blocksWorld;
    COMM_SESSION.allreduce(
        blocksSecondary.data(), blocksWorld.data(), windowSizes.size(), mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    blocksSecondary = blocksWorld;
    TLLM_LOG_DEBUG(
        "[RANK %d] After mpi::MpiOp::MIN reduction: window sizes %s / primary blocks %s / secondary blocks %s", rank,
        vec2str(windowSizes).c_str(), vec2str(blocksPrimary).c_str(), vec2str(blocksSecondary).c_str());
}

/// Reserve static memory for recurrent-state blocks (maxBatchSize blocks, scaled by the number of linear-attention
/// layers since allRecurrentStatesBytes is per-layer). Subtracts the reservation from @p allottedPrimaryMemBytes
/// (in/out) and routes the recurrent-states window's share of the dynamic budget to the full-attention window.
void reserveStaticHybridLinearStates(LinearAttentionMetadata const& linearAttentionMetadata,
    std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers, SizeType32 maxBatchSize,
    uint64_t& allottedPrimaryMemBytes, std::map<SizeType32, float>& windowSizeToShare)
{
    for (auto const& [windowSize, _] : windowSizeToLayers)
    {
        windowSizeToShare[windowSize] = 1.0f;
    }
    windowSizeToShare[LinearAttentionMetadata::kRecurrentStates] = 0.0f;

    auto const numLinearLayers
        = static_cast<uint64_t>(windowSizeToLayers.at(LinearAttentionMetadata::kRecurrentStates).size());
    auto const staticMemoryBytes = static_cast<uint64_t>(maxBatchSize) * numLinearLayers
        * static_cast<uint64_t>(linearAttentionMetadata.allRecurrentStatesBytes);
    TLLM_CHECK_WITH_INFO(allottedPrimaryMemBytes > staticMemoryBytes,
        "Not enough memory to allocate static cache for linear attention states. Required: %0.2f GiB, "
        "Allotted: %0.2f GiB",
        staticMemoryBytes / static_cast<double>(1 << 30), allottedPrimaryMemBytes / static_cast<double>(1 << 30));
    allottedPrimaryMemBytes -= staticMemoryBytes;
    TLLM_LOG_INFO(
        "Hybrid model with linear attention (mamba) detected. Reserving %0.2f GiB of GPU memory for static "
        "allocation of linear attention states, leaving %0.2f GiB for dynamically allocated blocks for KV cache.",
        staticMemoryBytes / static_cast<double>(1 << 30), allottedPrimaryMemBytes / static_cast<double>(1 << 30));
    if (std::getenv("TRTLLM_WINDOW_SIZE_SHARES"))
    {
        TLLM_LOG_INFO("TRTLLM_WINDOW_SIZE_SHARES is ignored in this case");
    }
}

/// Compute each window's share of the dynamic primary-memory budget.
///
/// By default, all windows get equal shares. With TRTLLM_WINDOW_SIZE_SHARES, shares are taken from the env var
/// (e.g. "0.4,0.6") and normalized to sum to 1.
std::map<SizeType32, float> computeWindowSizeShares(
    std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers)
{
    std::map<SizeType32, float> windowSizeToShare;
    // By default, we allocate equal proportion shares of memory for all
    // window sizes (see the else case). With TRTLLM_WINDOW_SIZE_SHARES,
    // we can override this behavior to adjust the memory share of each
    // window size. For example, if we have window size of [512, 32768],
    // then setting TRTLLM_WINDOW_SIZE_SHARES=0.4,0.6 will be allocating
    // 40% of the memory to window size 512 and 60% of the memory to window
    // size 32768.
    if (auto envStr = std::getenv("TRTLLM_WINDOW_SIZE_SHARES"))
    {
        std::stringstream ss(envStr);
        std::vector<float> shares;
        float share;
        while (ss >> share)
        {
            shares.push_back(share);
            if (ss.peek() == ',')
                ss.ignore();
        }

        TLLM_CHECK_WITH_INFO(shares.size() == windowSizeToLayers.size(),
            "Number of shares in TRTLLM_WINDOW_SIZE_SHARES (%ld) must match number of window sizes (%ld)",
            shares.size(), windowSizeToLayers.size());
        float sumShares = 0.0f;
        for (auto s : shares)
        {
            TLLM_CHECK_WITH_INFO(0.0f <= s && s <= 1.0f, "Shares must be in value range [0,1], got %f", s);
            sumShares += s;
        }
        TLLM_CHECK_WITH_INFO(sumShares > 0.0f, "Sum of shares must be > 0.");
        // Normalize shares to 1.0
        for (auto& s : shares)
        {
            s /= sumShares;
        }
        size_t i = 0;
        for (auto const& [windowSize, _] : windowSizeToLayers)
        {
            windowSizeToShare[windowSize] = shares[i++];
        }
    }
    else
    {
        // NOTE: Righteously, blocks allocated should be proportional with
        // regard to window size. Currently, we are first allocating identical
        // number of blocks for all layers to achieve identical performance.
        for (auto const& [windowSize, _] : windowSizeToLayers)
        {
            windowSizeToShare[windowSize] = 1.0f / windowSizeToLayers.size();
        }
    }
    return windowSizeToShare;
}
} // namespace

BlocksPerWindow BaseKVCacheManager::calculateMaxNumBlocks(executor::KvCacheConfig const& config,
    nvinfer1::DataType dtype, std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, WorldConfig const& worldConfig,
    std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers, uint64_t allottedPrimaryMemBytes,
    uint64_t allottedSecondaryMemBytes, size_t extraCostMemory, SizeType32 kvFactor, SizeType32 maxBatchSize,
    std::optional<LinearAttentionMetadata> const& linearAttentionMetadata)
{
    TLLM_LOG_DEBUG("Calculating max num blocks: {.allottedPrimaryMemBytes=%" PRIu64
                   ", .allottedSecondaryMemBytes=%" PRIu64 "}",
        allottedPrimaryMemBytes, allottedSecondaryMemBytes);

    if (config.getMaxTokens().has_value() && windowSizeToLayers.size() > 1)
    {
        TLLM_LOG_WARNING(
            "Setting maxTokens when using Variable Sliding Window Attention is a strange concept, as it limits "
            "the number of max tokens *per window size* [limiting the sum of all window sizes is even stranger]. "
            "Anticipating the effects of this requires quite a complex calculation, and it probably isn't the "
            "configuration you meant to use.");
    }

    std::map<SizeType32, SizeType32> cacheSizeBytesPerTokenPerWindow;
    for (auto const& [windowSize, managedLayers] : windowSizeToLayers)
    {
        auto const cacheSizePerToken = LinearAttentionMetadata::hasLinearCache(windowSize)
            ? 1
            : BaseKVCacheManager::calculateCacheSizePerTokenForSingleWindowSize(
                numKvHeadsPerLayer, sizePerHead, managedLayers, kvFactor);
        auto const cacheSizeBytesPerToken = cacheSizePerToken * BufferDataType(dtype).getSize();
        cacheSizeBytesPerTokenPerWindow[windowSize] = cacheSizeBytesPerToken;
    }

    // When the model is mamba hybrid (i.e. has only 2 windows sizes: max_seq_len and
    // LinearAttentionMetadata::kRecurrentStates), and block reuse is disabled, we can allocate static memory for the
    // linear attention states - max_batch_size blocks.
    bool const isStaticHybridNoReuseModel = windowSizeToLayers.size() == 2
        && windowSizeToLayers.count(LinearAttentionMetadata::LinearCacheType::kRecurrentStates) == 1
        && linearAttentionMetadata.has_value() && !config.getEnableBlockReuse();
    bool const isVSWA = cacheSizeBytesPerTokenPerWindow.size() > 1;

    TLLM_LOG_DEBUG("extraCostMemory [Gib]: %0.2f", extraCostMemory / static_cast<double>(1 << 30));
    allottedPrimaryMemBytes = allottedPrimaryMemBytes - extraCostMemory;

    auto windowSizeToShare = computeWindowSizeShares(windowSizeToLayers);

    if (isStaticHybridNoReuseModel)
    {
        reserveStaticHybridLinearStates(
            *linearAttentionMetadata, windowSizeToLayers, maxBatchSize, allottedPrimaryMemBytes, windowSizeToShare);
    }

    auto const calculatePrimaryBlocks
        = [&](SizeType32 windowSize, double windowSizeShare, SizeType32 cacheSizeBytesPerToken)
    {
        TLLM_LOG_DEBUG("windowSizeShare: %lf, cacheSizeBytesPerToken: %d", windowSizeShare, cacheSizeBytesPerToken);
        auto memoryBudget = static_cast<uint64_t>(allottedPrimaryMemBytes * windowSizeShare);
        if (LinearAttentionMetadata::hasRecurrentStatesCache(windowSize))
        {
            TLLM_CHECK_WITH_INFO(linearAttentionMetadata.has_value(),
                "Linear attention metadata must be provided when linear attention is used.");
            return isStaticHybridNoReuseModel ? maxBatchSize
                                              : linearAttentionMetadata->calcMaxMemoryBlocks(windowSize, tokensPerBlock,
                                                  memoryBudget, windowSizeToLayers.at(windowSize).size());
        }
        auto maxTokens = static_cast<uint64_t>(memoryBudget / cacheSizeBytesPerToken);

        // kv_cache_config.max_tokens is not effective in VSWA scheme
        if (config.getMaxTokens().has_value() && !isVSWA)
        {
            auto const maxTokensFromConfig = static_cast<uint64_t>(config.getMaxTokens().value());
            if (maxTokensFromConfig < maxTokens)
            {
                TLLM_LOG_DEBUG("Maximum kv-cache token overridden by configuration as '%ld'.", maxTokensFromConfig);
                maxTokens = std::min(maxTokensFromConfig, maxTokens);
            }
        }
        TLLM_LOG_DEBUG("Primary maxTokens for windowSize %d: %ld", windowSize, maxTokens);
        return static_cast<SizeType32>(tc::ceilDiv(maxTokens, tokensPerBlock));
    };

    auto const calculateSecondaryBlocks
        = [&](SizeType32 windowSize, double windowSizeShare, SizeType32 cacheSizeBytesPerToken)
    {
        auto memoryBudget = static_cast<uint64_t>(allottedSecondaryMemBytes * windowSizeShare);
        if (LinearAttentionMetadata::hasLinearCache(windowSize))
        {
            if (isStaticHybridNoReuseModel && memoryBudget > 0)
            {
                TLLM_LOG_WARNING("It's meaningless to allocate secondary memory when block reuse is disabled.");
            }
            TLLM_CHECK_WITH_INFO(linearAttentionMetadata.has_value(),
                "Linear attention metadata must be provided when linear attention is used.");
            return linearAttentionMetadata->calcMaxMemoryBlocks(
                windowSize, tokensPerBlock, memoryBudget, windowSizeToLayers.at(windowSize).size());
        }
        auto maxTokensSecondary = static_cast<SizeType32>(memoryBudget / cacheSizeBytesPerToken);
        return std::max(0, maxTokensSecondary / tokensPerBlock);
    };

    std::vector<SizeType32> blocksPrimary;
    std::vector<SizeType32> blocksSecondary;
    for (auto const& [windowSize, managedLayers] : windowSizeToLayers)
    {
        auto const cacheSizeBytesPerToken = cacheSizeBytesPerTokenPerWindow.at(windowSize);
        auto const windowSizeShare = windowSizeToShare.at(windowSize);
        auto const blocksInPrimaryPool = calculatePrimaryBlocks(windowSize, windowSizeShare, cacheSizeBytesPerToken);
        TLLM_LOG_DEBUG(
            "Number of blocks in KV cache primary pool for windowSize %d: %d", windowSize, blocksInPrimaryPool);
        auto const blocksInSecondaryPool
            = calculateSecondaryBlocks(windowSize, windowSizeShare, cacheSizeBytesPerToken);
        TLLM_LOG_DEBUG(
            "Number of blocks in KV cache secondary pool for windowSize %d: %d", windowSize, blocksInSecondaryPool);
        blocksPrimary.push_back(blocksInPrimaryPool);
        blocksSecondary.push_back(blocksInSecondaryPool);
    }

    std::vector<SizeType32> windowSizes;
    windowSizes.reserve(windowSizeToLayers.size());
    for (auto const& [k, _] : windowSizeToLayers)
    {
        windowSizes.push_back(k);
    }
    // windowSizes is sorted thanks to windowSizeToLayers being a std::map.
    synchronizeBlockCountsAcrossRanks(worldConfig, windowSizes, blocksPrimary, blocksSecondary);

    BlocksPerWindow windowSizeToBlocks;
    TLLM_LOG_INFO("Blocks per window size:");
    for (size_t i = 0; i < windowSizes.size(); ++i)
    {
        auto const windowSize = windowSizes.at(i);
        auto const primaryBlocks = blocksPrimary.at(i);
        auto const secondayBlocks = blocksSecondary.at(i);
        TLLM_LOG_INFO(
            "[windowSize=%d] {.primaryBlocks=%d, .secondayBlocks=%d}", windowSize, primaryBlocks, secondayBlocks);
        windowSizeToBlocks[windowSize] = {primaryBlocks, secondayBlocks};
    }
    return windowSizeToBlocks;
}

void KVCacheManager::removeToken(RequestIdType requestId)
{
    // TODO: add streamLLM support
    auto& sequence = getSequence(requestId);
    if (sequence.getNumTokens() == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(sequence.getBeamWidth() == 1, "[kv cache manager] removeToken does not support beamWidth > 1");
    sequence.removeTokens(1);
    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        SizeType32 const tokensInWindow = sequence.getNumTokens() % windowSize;
        if (tokensInWindow % getTokensPerBlock() == 0)
        {
            mBlockManager.releaseLastBlock(sequence, windowSize);
        }
    }
}

void KVCacheManager::rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths)
{
    // Check if the sequence still exists before rewinding
    // In overlap mode with MTP, the request may have been terminated and removed
    // from mSequences before rewindKVCache is called
    {
        std::scoped_lock lck(mSequencesMtx);
        if (mSequences.find(requestId) == mSequences.end())
        {
            TLLM_LOG_DEBUG("Request %lu has already been removed from KV cache manager, skipping rewind", requestId);
            return;
        }
    }

    for (SizeType32 si = 0; si < rewindLengths; ++si)
    {
        removeToken(requestId);
    }
}

GenerationRequest const& KVCacheManager::getSequence(RequestIdType requestId) const
{
    auto lck = std::scoped_lock(mSequencesMtx);
    return mSequences.at(requestId);
}

GenerationRequest& KVCacheManager::getSequence(RequestIdType requestId)
{
    auto lck = std::scoped_lock(mSequencesMtx);
    return mSequences.at(requestId);
}

SizeType32 KVCacheManager::getTokenCount(RequestIdType requestId) const
{
    return getSequence(requestId).getNumTokens();
}

SizeType32 BaseKVCacheManager::getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock)
{
    auto const sinkTokensInLastBlock = sinkTokenLen % tokensPerBlock;
    auto const sinkBubbleLength = sinkTokensInLastBlock == 0 ? 0 : tokensPerBlock - sinkTokensInLastBlock;
    return sinkBubbleLength;
}

std::vector<std::vector<SizeType32>> const& KVCacheManager::getCacheBlockIds(
    RequestIdType requestId, SizeType32 windowSize) const
{
    return getSequence(requestId).getCacheBlockIds(windowSize);
}

std::vector<std::vector<std::vector<SizeType32>>> KVCacheManager::getBatchCacheBlockIds(
    std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const
{
    std::vector<std::vector<std::vector<SizeType32>>> result{};
    result.reserve(requestIds.size());
    for (auto const& requestId : requestIds)
    {
        auto const& sequence = getSequence(requestId);
        result.emplace_back(sequence.getCacheBlockIds(windowSize));
    }
    return result;
}

std::optional<KVCacheBlock::IdType> KVCacheManager::getLastBlockId(LlmRequest::RequestIdType requestId) const
{
    auto const& seq = getSequence(requestId);
    // Use the first window size
    auto firstWindowSize = mBlockManager.getFirstWindowSize();
    if (firstWindowSize == 0)
    {
        return std::nullopt;
    }
    auto const& perBeam = seq.getCacheBlockIds(firstWindowSize);
    if (perBeam.empty() || perBeam[0].empty())
    {
        return std::nullopt;
    }
    return perBeam[0].back();
}

runtime::ITensor::SharedPtr KVCacheManager::getUniquePrimaryPool() const
{
    TLLM_CHECK_WITH_INFO(mBlockManager.getWindowSizesMetadata().size() == 1,
        "getUniquePrimaryPool is only supported for a single window size");
    return mBlockManager.getPrimaryPool(0);
}

runtime::ITensor::SharedPtr KVCacheManager::getPrimaryPool(SizeType32 layer_idx) const
{
    return mBlockManager.getPrimaryPool(mBlockManager.getLayerPoolIdx(layer_idx));
}

runtime::ITensor::SharedPtr KVCacheManager::getIndexerKCachePool() const
{
    return mIndexerKCachePoolPointers;
}

SizeType32 KVCacheManager::getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const
{
    auto minMaxBatchSizeAllWindows = std::numeric_limits<SizeType32>::max();
    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        auto const blockRequirementsPerSequence = KVCacheManager::calculateMaxBlockRequirements(
            inputLength, outputLength, mSinkBlockTokenLength, windowSize, mMaxBeamWidth, mTokensPerBlock);
        auto const maxBatchSizeWindow = metadata.allottedPrimaryBlocks / blockRequirementsPerSequence;
        // The window with the *smallest* max batch size is the limiting factor
        // Hence, the std::*min* of all the max batch sizes is chosen
        minMaxBatchSizeAllWindows = std::min(minMaxBatchSizeAllWindows, maxBatchSizeWindow);
    }
    return minMaxBatchSizeAllWindows;
}

SizeType32 KVCacheManager::calculateMaxBlockRequirementsPerBeam(
    SizeType32 sequenceLength, SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 tokensPerBlock)
{
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const actualSeqLen = std::min(sequenceLength, windowSize);
    auto actualMaxTokenNum = actualSeqLen + sinkBubbleLength;
    auto numBlocks = tc::ceilDiv(actualMaxTokenNum, tokensPerBlock);
    if (sequenceLength > windowSize)
    {
        numBlocks += kSWAExtraBlock;
    }
    return numBlocks;
}

SizeType32 KVCacheManager::calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
    SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 beamWidth, SizeType32 tokensPerBlock)
{
    // We split beam width > 1, as it introduces a lot of complexity.
    auto const wholeSequenceLength = inputLength + outputLength;
    if (beamWidth == 1)
    {
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
            wholeSequenceLength, sinkTokenLength, windowSize, tokensPerBlock);
    }

    if (windowSize <= outputLength)
    {
        // We at most will need outputLength of distinct blocks for SWA
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
                   outputLength, sinkTokenLength, windowSize, tokensPerBlock)
            * beamWidth;
    }

    // Otherwise, we calculate how many tokens will be in output blocks.
    auto const effectiveAttentionWindow = std::min(windowSize, wholeSequenceLength);
    auto const numContextTokensInAttentionWindow
        = effectiveAttentionWindow - outputLength; // This is positive because we handled the other case above.
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const numContextBlocks = (numContextTokensInAttentionWindow + sinkBubbleLength) / tokensPerBlock;
    auto const leftoverContextToken = numContextTokensInAttentionWindow - numContextBlocks * tokensPerBlock;
    auto numOutputBlocks = tc::ceilDiv(outputLength + leftoverContextToken, tokensPerBlock);
    if (wholeSequenceLength > windowSize)
    {
        numOutputBlocks += kSWAExtraBlock;
    }
    return numContextBlocks + numOutputBlocks * beamWidth;
}

[[nodiscard]] SizeType32 KVCacheManager::calculateMaxAttentionWindow(SizeType32 inputLength, SizeType32 outputLength,
    SizeType32 sinkTokenLength, SizeType32 blockCapacity, SizeType32 beamWidth, SizeType32 tokensPerBlock)
{
    // The function that gives the number of blocks required given an attention window is only linear by part. It is
    // however, monotonically increasing in the attention window. Therefore, we need to find in which part of the
    // function we are. First, are we in a case where not even the entire output will fit?
    auto const outputBlockRequirements
        = calculateMaxBlockRequirements(0, outputLength, sinkTokenLength, outputLength, beamWidth, tokensPerBlock);
    if (outputBlockRequirements > blockCapacity)
    {
        return (blockCapacity / beamWidth) * tokensPerBlock;
    }

    // Otherwise, we need to determine how many context tokens we can fit on top of the output tokens. First, there
    // are a few context tokens we might be able to fit 'for free' because the output is not a multiple of the
    // number of tokens per block.
    auto const leftoverBlockCapacity = blockCapacity - outputBlockRequirements;
    return std::min(outputLength + leftoverBlockCapacity * tokensPerBlock, inputLength + outputLength);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
""",
    'key-files/cpp/tensorrt_llm/executor/cache_transmission/cacheSplitConcat.cu': r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cacheSplitConcat.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{
inline bool isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}
} // namespace

static inline int computeDimsPerHead(kv_cache::CacheState const& cacheState, bool isIndexerKCache)
{
    if (!isIndexerKCache)
    {
        return cacheState.getModelConfig().mSizePerHead;
    }
    auto const dim = cacheState.getIndexerDimPerHead();
    auto const q = cacheState.getIndexerKCacheQuantBlockSize();
    // Scale bytes: dim / quantBlockSize * 4 (same for FP8 and FP4).
    // Data bytes: dim for FP8 (one byte per element), dim / 2 for FP4
    // (two packed E2M1 codes per byte).
    auto const scaleBytes = dim / q * 4;
    auto const dataBytes = cacheState.getIndexerKCacheUseFp4() ? dim / 2 : dim;
    return dataBytes + scaleBytes;
}

int getBlockNumAccountingForCP(int cpRank, int cpSize, int numTotalBlocks)
{
    TLLM_CHECK(cpRank >= 0 && cpRank < cpSize);
    if (cpSize == 1)
    {
        return numTotalBlocks;
    }
    // Blocks are distributed among CP ranks in a round-robin fashion. When the number of blocks is not
    // divisible by cpSize, the lowest-indexed CP ranks each receive one extra block.
    int numBlocksCurrRank = numTotalBlocks / cpSize;
    if (numTotalBlocks % cpSize > cpRank)
    {
        numBlocksCurrRank++;
    }
    return numBlocksCurrRank;
}

// inputBlockNums: [outputBlockNum, inputRanks.size]
// [PP, TP]
// Core implementation for computing target ranks info for data parallelism.
// Takes explicit layer vectors and head counts so it can be reused for both KV and RNN cache.
TargetRanksInfo TargetRanksInfoForDPImpl(kv_cache::CacheState const& peerCacheState,
    kv_cache::CacheState const& selfCacheState, int selfRank, std::vector<SizeType32> const& peerNumLayerPerPP,
    std::vector<SizeType32> const& selfNumLayerPerPP, SizeType32 peerNbHeadsPerLayer, SizeType32 selfNbHeadsPerLayer)
{
    auto const& peerParConfig = peerCacheState.getParallelConfig();
    auto const& selfParConfig = selfCacheState.getParallelConfig();

    auto const peerPPNum = peerParConfig.mPipelineParallelism;
    auto const selfPPNum = selfParConfig.mPipelineParallelism;
    auto const peerTPNum = peerParConfig.mTensorParallelism;
    auto const selfTPNum = selfParConfig.mTensorParallelism;
    auto const peerCPNum = peerParConfig.mContextParallelism;
    auto const selfCPNum = selfParConfig.mContextParallelism;

    auto const selfCPRank = selfRank % selfCPNum;
    auto const selfTPRank = (selfRank % (selfTPNum * selfCPNum)) / selfCPNum;
    auto const selfPPRank = selfRank / (selfTPNum * selfCPNum);

    int peerPPRankStart = 0;
    int mDomainPPSize = 1;
    int peerPPRankEnd = 0;

    if (selfNumLayerPerPP[selfPPRank] == 0)
    {
        return TargetRanksInfo{.mDomainPPSize = 0,
            .mDomainTPSize = 0,
            .mDomainCPSize = 0,
            .mIRanks = {}, // caller should handle this case. No transfer needed.
            .mDupHeadFactor = 1,
            .mPeerDupHeadFactor = 1,
            .mPeerLayerNumInDomainPP = {}};
    }

    TLLM_CHECK(peerNumLayerPerPP.size() == peerPPNum);
    TLLM_CHECK(selfNumLayerPerPP.size() == selfPPNum);
    int selfStartLayerId = 0;
    // global start layer id for selfPPrank, which is the sum of the layer num of the previous PP ranks.
    // compute the target PP ranks and layer num need to be fetched from each target PP rank, according to [global start
    // layer id, global end layer id)

    for (int ppRank = 0; ppRank < selfPPRank; ppRank++)
    {
        selfStartLayerId += selfNumLayerPerPP[ppRank];
    }
    int selfEndLayerId = selfStartLayerId + selfNumLayerPerPP[selfPPRank];

    int prePeerPPLayerId = 0;
    std::vector<int> targetPeerPPRanks;
    std::vector<int> targetPeerPPLayerNum;
    for (int ppRank = 0; ppRank < peerPPNum; ppRank++)
    {
        int peerPPStartLayerId = prePeerPPLayerId;
        int peerPPEndLayerId = peerPPStartLayerId + peerNumLayerPerPP[ppRank];

        prePeerPPLayerId += peerNumLayerPerPP[ppRank];

        if (selfStartLayerId < peerPPEndLayerId && selfEndLayerId > peerPPStartLayerId)
        {
            targetPeerPPRanks.push_back(ppRank);
            int layerNumInDomainPP
                = std::min(peerPPEndLayerId, selfEndLayerId) - std::max(peerPPStartLayerId, selfStartLayerId);
            targetPeerPPLayerNum.push_back(layerNumInDomainPP);
        }
    }
    mDomainPPSize = static_cast<int>(targetPeerPPRanks.size());
    peerPPRankStart = targetPeerPPRanks.front();
    peerPPRankEnd = peerPPRankStart + mDomainPPSize;
    TLLM_CHECK(targetPeerPPLayerNum.size() == mDomainPPSize);

    int targetPeerPpLayerNumSum = std::accumulate(targetPeerPPLayerNum.begin(), targetPeerPPLayerNum.end(), 0);
    TLLM_CHECK(targetPeerPpLayerNumSum == selfNumLayerPerPP[selfPPRank]);

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "selfPPRank:%d,selfPPNum:%d,peerPPNum:%d,selfTPNum:%d,peerTPNum:%d,peerPPRankStart:%d,peerPPRankEnd:%d",
        selfPPRank, selfPPNum, peerPPNum, selfTPNum, peerTPNum, peerPPRankStart, peerPPRankEnd);
    int peerTPRankStart = 0;
    int mDomainTPSize = 1;
    int peerTPRankEnd = 0;

    int const peerDpRank = peerParConfig.mEnableAttentionDP ? peerParConfig.mDPrank : 0;
    int const selfTPSizePerDPGroup = selfParConfig.mEnableAttentionDP ? selfTPNum / selfParConfig.mDPsize : selfTPNum;
    int const peerTPSizePerDPGroup = peerParConfig.mEnableAttentionDP ? peerTPNum / peerParConfig.mDPsize : peerTPNum;

    int const selfTPrankInDPGroup = selfTPRank % selfTPSizePerDPGroup;
    for (auto val : {peerTPSizePerDPGroup, selfTPSizePerDPGroup})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }
    if (selfTPSizePerDPGroup <= peerTPSizePerDPGroup)
    {
        mDomainTPSize = peerTPSizePerDPGroup / selfTPSizePerDPGroup;
        peerTPRankStart = selfTPrankInDPGroup * mDomainTPSize + peerDpRank * peerTPSizePerDPGroup;
        peerTPRankEnd = peerTPRankStart + mDomainTPSize;
    }
    else
    {
        peerTPRankStart
            = selfTPrankInDPGroup / (selfTPSizePerDPGroup / peerTPSizePerDPGroup) + peerDpRank * peerTPSizePerDPGroup;
        peerTPRankEnd = peerTPRankStart + mDomainTPSize;
    }

    int mDomainCPSize = 1;
    int peerCPRankStart = 0;
    int peerCPRankEnd = 0;
    for (auto val : {peerCPNum, selfCPNum})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }
    if (selfCPNum <= peerCPNum)
    {
        mDomainCPSize = peerCPNum / selfCPNum;
        peerCPRankStart = selfCPRank * mDomainCPSize;
        peerCPRankEnd = (selfCPRank + 1) * mDomainCPSize;
    }
    else
    {
        peerCPRankStart = selfCPRank / (selfCPNum / peerCPNum);
        peerCPRankEnd = peerCPRankStart + mDomainCPSize;
    }

    std::vector<int> retRanks;
    for (int i = peerCPRankStart; i < peerCPRankEnd; i++)
    {
        for (int j = peerTPRankStart; j < peerTPRankEnd; j++)
        {
            for (int k = peerPPRankStart; k < peerPPRankEnd; k++)
            {
                // Rank formula: ppRank * (tpNum * cpNum) + tpRank * cpNum + cpRank.
                int irank = (k * peerTPNum * peerCPNum) + (j * peerCPNum) + i;
                retRanks.push_back(irank);
            }
        }
    }

    int mDupHeadFactor = 1;
    int mPeerDupHeadFactor = 1;

    if (selfNbHeadsPerLayer * selfTPSizePerDPGroup > peerNbHeadsPerLayer * peerTPSizePerDPGroup)
    {
        mDupHeadFactor = (selfNbHeadsPerLayer * selfTPSizePerDPGroup) / (peerNbHeadsPerLayer * peerTPSizePerDPGroup);
    }
    if (peerNbHeadsPerLayer * peerTPSizePerDPGroup > selfNbHeadsPerLayer * selfTPSizePerDPGroup)
    {
        mPeerDupHeadFactor
            = (peerNbHeadsPerLayer * peerTPSizePerDPGroup) / (selfNbHeadsPerLayer * selfTPSizePerDPGroup);
    }

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "mDomainPPSize:%d, mDomainTPSize:%d, mDupHeadFactor:%d, mPeerDupHeadFactor:%d, selfPPRank:%d, selfPPNum:%d, "
        "peerPPNum:%d, selfTPNum:%d, peerTPNum:%d, selfTPSizePerDPGroup:%d, peerTPSizePerDPGroup:%d, "
        "selfNbHeadsPerLayer:%d, peerNbHeadsPerLayer:%d, selfTPrankInDPGroup:%d, peerDpRank:%d, selfRank:%d",
        mDomainPPSize, mDomainTPSize, mDupHeadFactor, mPeerDupHeadFactor, selfPPRank, selfPPNum, peerPPNum, selfTPNum,
        peerTPNum, selfTPSizePerDPGroup, peerTPSizePerDPGroup, selfNbHeadsPerLayer, peerNbHeadsPerLayer,
        selfTPrankInDPGroup, peerDpRank, selfRank);

    auto vector_to_string = [](std::vector<int> const& vec)
    {
        std::stringstream ss;
        for (auto val : vec)
        {
            ss << val << ",";
        }
        return ss.str();
    };
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "retRanks:%s , targetPeerPPLayerNum:%s",
        vector_to_string(retRanks).c_str(), vector_to_string(targetPeerPPLayerNum).c_str());
    return {mDomainPPSize, mDomainTPSize, mDomainCPSize, std::move(retRanks), mDupHeadFactor, mPeerDupHeadFactor,
        std::move(targetPeerPPLayerNum)};
}

TargetRanksInfo targetIRanks(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    auto const& peerHeads = peerCacheState.getModelConfig().mNbKvHeadsPerLayer;
    auto const& selfHeads = selfCacheState.getModelConfig().mNbKvHeadsPerLayer;
    return TargetRanksInfoForDPImpl(peerCacheState, selfCacheState, selfRank,
        peerCacheState.getParallelConfig().mAttentionLayerNumPerPP,
        selfCacheState.getParallelConfig().mAttentionLayerNumPerPP, peerHeads.empty() ? 0 : peerHeads[0],
        selfHeads.empty() ? 0 : selfHeads[0]);
}

TargetRanksInfo targetIRanksForRnn(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    auto targetInfo = TargetRanksInfoForDPImpl(peerCacheState, selfCacheState, selfRank,
        peerCacheState.getRnnCacheState().mLayerNumPerPP, selfCacheState.getRnnCacheState().mLayerNumPerPP,
        peerCacheState.getRnnModelConfig().mNumHeads, selfCacheState.getRnnModelConfig().mNumHeads);
    targetInfo.mDupHeadFactor = 1;
    targetInfo.mPeerDupHeadFactor = 1; // RNN cache does not have head duplication.
    return targetInfo;
}

template <typename T>
struct BlockInfo
{
    T* data;
    int startTokenId;
    int tokensPerBlock;
    int startHeadId;
    int headsPerBlock;
    int startLayerId;
    int layersPerBlock;
    int dimsPerHead;
    size_t offset; // (data-offset)[idx]

    __forceinline__ __device__ __host__ T* getKblockPtr(int layerid) const
    {
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVblockPtr(int layerid) const
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getKDimsPtr(int layerid, int headid, int tokenid)
    {
        return getKblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getKDimsPtr(int layerid, int headid, int tokenid) const
    {
        return getKblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVDimsPtr(int layerid, int headid, int tokenid)
    {
        return getVblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getVDimsPtr(int layerid, int headid, int tokenid) const
    {
        return getVblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "{data ptr: " << data << ", startTokenId: " << startTokenId << ", tokensPerBlock: " << tokensPerBlock
           << ", startHeadId: " << startHeadId << ", headsPerBlock: " << headsPerBlock
           << ", startLayerId:" << startLayerId << ", layersPerBlock: " << layersPerBlock
           << ", dimsPerHead: " << dimsPerHead << ", offset: " << offset << "}";
        return ss.str();
    }
};

// Reference to blockPtr

// Block shape: [numHeads, numTokens, dimsPerHead]
// CacheBlock shape: [numLayers, 2, mBlockSize]
// Note: mBlockSize refers to the size of each block

// Handling key and value copying
// Note: k and v are not stored contiguously in memory

__forceinline__ __device__ int getInputBlockId(int outputBlockId, int headId, int layerId, int inputBlockNumEachOutput,
    int headNumPerBlock, int layerNumPerBlock, int headNumInputModel, int layerNumInputModel)
{
    int const offset = outputBlockId * inputBlockNumEachOutput;
    int const layerOffset = layerId / layerNumPerBlock;
    int const headOffset = headId / headNumPerBlock;
    int const headBlockNum = headNumInputModel / headNumPerBlock;
    return offset + layerOffset * headBlockNum + headOffset;
}

// subWarpSize * subWarpGroupSize
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitAndConcatBlocksKernel(BlockInfo<T> const* iBlockInfo, BlockInfo<T>* oBlockInfo, int iBlockNum,
    int iNumBlockEachO, int oBlockNum, int headNumInputModel, int layerNumInputModel, int iHeadsPerBlock,
    int iLayersPerBlock)
{
    // blockDim.y corresponds to the number of output blocks
    // blockDim.x corresponds to the number of layers

    // Warp-level parallelism spans heads * tokens
    // Thread-level parallelism spans dimsPerHead

    // input_id can be derived from output_id, layer_id, and head_id
    // Total number of CUDA blocks = numLayers * outputBlockNum

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;

#pragma unroll 1
    for (int oBlockId = blockIdx.y; oBlockId < oBlockNum; oBlockId += gridDim.y)
    {
        int oLayerNum = oBlockInfo[oBlockId].layersPerBlock;
        int headNum = oBlockInfo[oBlockId].headsPerBlock;
        int tokenNum = oBlockInfo[oBlockId].tokensPerBlock;
        int dimsPerHead = oBlockInfo[oBlockId].dimsPerHead;
#pragma unroll 1

        for (int layerid = blockIdx.x; layerid < oLayerNum; layerid += gridDim.x)
        {
#pragma unroll 1
            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {
                int const targetHeadId = oBlockInfo[oBlockId].startHeadId + headId;
                int const targetLayerId = oBlockInfo[oBlockId].startLayerId + layerid;

                int const iBlockId = getInputBlockId(oBlockId, targetHeadId, targetLayerId, iNumBlockEachO,
                    iHeadsPerBlock, iLayersPerBlock, headNumInputModel, layerNumInputModel);
                int const iLayerId = targetLayerId % iLayersPerBlock;
                int const iHeadId = targetHeadId % iHeadsPerBlock;
#pragma unroll 1
                for (int tokenId = subWarpIdInGroup; tokenId < tokenNum; tokenId += subWarpNumInGroup)
                {
                    T* oKPtr = oBlockInfo[oBlockId].getKDimsPtr(layerid, headId, tokenId);
                    T const* iKPtr = iBlockInfo[iBlockId].getKDimsPtr(iLayerId, iHeadId, tokenId);
                    T* oVPtr = oBlockInfo[oBlockId].getVDimsPtr(layerid, headId, tokenId);
                    T const* iVPtr = iBlockInfo[iBlockId].getVDimsPtr(iLayerId, iHeadId, tokenId);
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        common::copy<vecSizeByte>(iKPtr + channelId, oKPtr + channelId);
                        common::copy<vecSizeByte>(iVPtr + channelId, oVPtr + channelId);
                    }
                }
            }
        }
    }
}

template <typename T>
void concatKVCache(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum, std::vector<int> const& inputRanks,
    kv_cache::CacheState const& iCacheState, runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRank,
    kv_cache::CacheState const& oCacheState, runtime::BufferManager const& bufferManager)

{
    TLLM_CHECK_WITH_INFO(!inputRanks.empty(), "inputRanks should not be empty.");
    TLLM_CHECK_WITH_INFO(inputBlockNum == outputBlockNum * inputRanks.size(),
        "inputBlockNum must equal outputBlockNum multiplied by the size of inputRanks.");
    TLLM_CHECK(inputRanks == targetIRanks(iCacheState, oCacheState, oRank).mIRanks);

    auto const& iParallelConfig = iCacheState.getParallelConfig();
    auto const& oParallelConfig = oCacheState.getParallelConfig();
    auto const& iModelConfig = iCacheState.getModelConfig();
    auto const& oModelConfig = oCacheState.getModelConfig();

    int const inputAllRankNum = iParallelConfig.mPipelineParallelism * iParallelConfig.mTensorParallelism;
    std::vector<BlockInfo<T>> blockInfos(outputBlockNum * inputAllRankNum + outputBlockNum);

    auto fillBlockInfo = [](kv_cache::CacheState const& cacheState, runtime::ITensor::SharedPtr buffer, int rank)
    {
        auto const& parallelConfig = cacheState.getParallelConfig();
        auto const& modelConfig = cacheState.getModelConfig();

        int const tpRank = rank % parallelConfig.mTensorParallelism;
        int const ppRank = rank / parallelConfig.mTensorParallelism;
        int const ppNum = parallelConfig.mPipelineParallelism;
        int const headsPerBlock = modelConfig.mNbKvHeadsPerLayer[0];
        int const layersPerBlock = modelConfig.mNbKvHeadsPerLayer.size() / ppNum;

        int const tokensPerBlock = modelConfig.mTokensPerBlock;
        int const dimsPerBlock = modelConfig.mSizePerHead;
        int const startHead = tpRank * headsPerBlock;
        int const startLayer = ppRank * layersPerBlock;

        constexpr int startTokenId = 0;
        auto* data = static_cast<T*>(buffer->data());
        return BlockInfo<T>{
            data, startTokenId, tokensPerBlock, startHead, headsPerBlock, startLayer, layersPerBlock, dimsPerBlock, 0};
    };
    // fill blcokInfo from CacheState and inputBlocks
    for (int oi = 0; oi < outputBlockNum; oi++)
    {
        int iRankNum = inputRanks.size();
        for (int i = 0; i < iRankNum; i++)
        {
            int iRank = inputRanks[i];
            blockInfos[oi * inputAllRankNum + iRank]
                = fillBlockInfo(iCacheState, inputBlocks[oi * iRankNum + i], iRank);
        }

        blockInfos[outputBlockNum * inputAllRankNum + oi] = fillBlockInfo(oCacheState, outputBlocks[oi], oRank);
    }
    runtime::BufferManager::IBufferPtr blockInfosDeviceBuffer
        = bufferManager.gpu(sizeof(BlockInfo<T>) * (blockInfos.size()), nvinfer1::DataType::kUINT8);
    bufferManager.copy((blockInfos.data()), *blockInfosDeviceBuffer, runtime::MemoryType::kCPU);

    BlockInfo<T>* iBlockInfoDevice = static_cast<BlockInfo<T>*>(blockInfosDeviceBuffer->data());
    BlockInfo<T>* oBlockInfoDevice = iBlockInfoDevice + outputBlockNum * inputAllRankNum;

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    constexpr int blockDimx = 128;

    int oPPNum = oParallelConfig.mPipelineParallelism;
    int iPPNum = iParallelConfig.mPipelineParallelism;
    unsigned int gridDimx = oModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const headsInputModel = iModelConfig.mNbKvHeadsPerLayer[0] * iParallelConfig.mTensorParallelism;
    int const layersInputModel = iModelConfig.mNbKvHeadsPerLayer.size();
    int const iHeadsPerBlock = iModelConfig.mNbKvHeadsPerLayer[0];
    int const iLayersPerBlock = iModelConfig.mNbKvHeadsPerLayer.size() / iPPNum;
    int const sizePerHead = oModelConfig.mSizePerHead;
    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 8:
    {
        splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
            break;
        }
    }
    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {

            splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 2>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 1>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
        }
        else
        {
            TLLM_THROW("concatKVCacheDispatch encountered an unsupported data type error.");
        }
    }
    }
}

void concatKVCacheDispatch(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum,
    std::vector<int> const& inputRanks, kv_cache::CacheState const& iCacheState,
    runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRanks, kv_cache::CacheState const& oCacheState,
    runtime::BufferManager const& bufferManager)
{
    auto dataType = outputBlocks[0]->getDataType();
    int dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
    {
        concatKVCache<int64_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 4:
    {
        concatKVCache<int32_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 2:
    {
        concatKVCache<int16_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }

    case 1:
    {
        concatKVCache<int8_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum, oRanks,
            oCacheState, bufferManager);
        break;
    }

    default:
    {
        TLLM_THROW("concatKVCacheDispatch encountered an unsupported data type error.");
    }
    }
}

nvinfer1::Dims makeShapeFromCacheState(kv_cache::CacheState const& cacheState)
{

    int64_t blockSize = static_cast<int64_t>(cacheState.getModelConfig().mNbKvHeadsPerLayer[0]
        * cacheState.getModelConfig().mTokensPerBlock * cacheState.getModelConfig().mSizePerHead);
    int PPNum = cacheState.getParallelConfig().mPipelineParallelism;
    return runtime::ITensor::makeShape(
        {static_cast<int64_t>(cacheState.getModelConfig().mNbKvHeadsPerLayer.size() / PPNum),
            cacheState.getAttentionConfig().mKvFactor, blockSize});
}

__device__ __forceinline__ void getLayerIdInDomainPPandRankInDomainPP(int layerId, int domainPPSize,
    uint64_t* prefixLayerNumDevPtr, int& layerIdInDomainPP, int& rankInDomainPP, int& layerNumInSpecPP)
{
    __shared__ int sharedLayerIdInDomainPP;
    __shared__ int sharedRankInDomainPP;
    __shared__ int sharedLayerNumInSpecPP;

#pragma unroll 1
    for (int ppRank = threadIdx.x; ppRank < domainPPSize; ppRank += blockDim.x)
    {
        if (layerId >= prefixLayerNumDevPtr[ppRank] && layerId < prefixLayerNumDevPtr[ppRank + 1])
        {
            sharedLayerIdInDomainPP = layerId - prefixLayerNumDevPtr[ppRank];
            sharedRankInDomainPP = ppRank;
            sharedLayerNumInSpecPP = prefixLayerNumDevPtr[ppRank + 1] - prefixLayerNumDevPtr[ppRank];
            break;
        }
    }

    __syncthreads();
    layerIdInDomainPP = sharedLayerIdInDomainPP;
    rankInDomainPP = sharedRankInDomainPP;
    layerNumInSpecPP = sharedLayerNumInSpecPP;
}

// MLA Head 1: One thread block per [(2), tokens, dimsPerHead]

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitKVCacheForMLAKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int domainPPSize,
    int domainTPSize, int domainCPSize, int kvFactor, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1

    for (int64_t blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
        // Blocks are distributed among CP ranks in a round-robin fashion.
        int const rankInDomainCP = blockId % domainCPSize;    // genCPRank to which this block belongs.
        int const blockIdInDomainCP = blockId / domainCPSize; // localBlockId on genCPRank.
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = 0; headId < headNum; headId++)
            {
                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                // Example to understand the indexing: Say PP=2 and CP=2 for generation while PP=1 and CP=1 for context:
                // genRank 0 is ppRank0, cpRank0
                // genRank 1 is ppRank0, cpRank1
                // genRank 2 is ppRank1, cpRank0
                // genRank 3 is ppRank1, cpRank1
                // However, the order of cache transmission (targetRankInfo.mIRanks) is {0, 2, 1, 3}. i.e. {{pp0cp0},
                // {pp1cp0}, {pp0cp1}, {pp1cp1}}. So, outputCaches of all ppRanks corresponding to a given cpRank are
                // grouped together. We do blockId % domainCPSize because blocks are distributed among cpRanks in a
                // round-robin fashion.
                int outputCacheIdx = rankInDomainCP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                int const headIdInDomainTP = headId;

                T* kOutputPtr = outputCachePtr
                    + blockIdInDomainCP
                        * (static_cast<int64_t>(layerNumInSpecPP * kvFactor * headNum * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;
                int const kvOffset = headNum * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpId; tokenId < tokensPerBlock; tokenId += subWarpNum)
                {
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += subWarpSize * numElePerThread)
                    {
#pragma unroll 1
                        for (int kvId = 0; kvId < kvFactor; kvId++)
                        {
                            common::copy<vecSizeByte>(
                                iKPtr + kvId * kvOffset + channelId, oKPtr + kvId * kvOffset + channelId);
                        }
                    }
                }
            }
        }
    }
}

// Block shape: [head, tokens, dimsPerHead]
// CacheBlock shape: [numLayers, 2, mBlockSize]
// Output split caches shape: [outputSplitCaches, numLayers, 2, head, tokensPerBlock, dimsPerHead]
// Note: The number of tokens can be large
// subWarpSize * subWarpGroupSize

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitKVCacheKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int domainPPSize,
    int domainTPSize, int headNumDomainTP, uint64_t* prefixLayerNumDevPtr, int domainCPSize)
{

    // layerNumDomainPP =  numLayers/domainPPSize

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;

    // Number of TP output caches (after duplicate head reduction).
    int const numTPOutputCaches = headNum / headNumDomainTP;

#pragma unroll 1

    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
        // Blocks are distributed among CP ranks in a round-robin fashion.
        int const rankInDomainCP = blockId % domainCPSize;    // genCPRank to which this block belongs.
        int const blockIdInDomainCP = blockId / domainCPSize; // localBlockId on genCPRank.

#pragma unroll 1

        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {

            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1

            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T const* vInputPtr = inputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                // Output cache ordering: [CP, reduced_TP, PP]
                int const outputCacheIdx = rankInDomainCP * (numTPOutputCaches * domainPPSize)
                    + headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                int const headIdInDomainTP = headId % headNumDomainTP;
                T* kOutputPtr = outputCachePtr
                    + static_cast<int64_t>(blockIdInDomainCP)
                        * (static_cast<int64_t>(layerNumInSpecPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;

                T* vOutputPtr = kOutputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1

                for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
                {
                    auto baseOffset = tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        auto offset = baseOffset + channelId;
                        common::copy<vecSizeByte>(kInputPtr + offset, kOutputPtr + offset);
                        common::copy<vecSizeByte>(vInputPtr + offset, vOutputPtr + offset);
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void getPosId(int const posLayerId, int const* blockNumPerWindow, int const* layersPerWindow,
    int windowNum, int domainPPSize, int& inputBlockId, int& inputLayerId, int& outputAllLayerOffset,
    int& outputPPOffset)
{

    int offset = 0;
    int inputTensorIdx = 0;
    int layerInInputTensor = 0;
    int tensorIdxOffset = 0;
    int prevOutputLayerOffset = 0;
    __shared__ int sharedInputBlockId;
    __shared__ int sharedInputLayerId;
    __shared__ int sharedOutputAllLayerOffset;
    __shared__ int sharedOutputPPOffset;

    if (threadIdx.x == 0)
    {
#pragma unroll 1
        for (int i = 0; i < windowNum; i++)
        {
            int blockNumInWindow = blockNumPerWindow[i];
            int layersInWindow = layersPerWindow[i];

            // tensorIdxOffset
            if (posLayerId < offset + (blockNumInWindow * layersInWindow))
            {
                int remaindLayerInWindow = posLayerId - offset;

                int layerInWindowQuotient = remaindLayerInWindow / layersInWindow;
                int layerInWindowRemainder = remaindLayerInWindow % layersInWindow;

                inputTensorIdx = layerInWindowQuotient + tensorIdxOffset;
                layerInInputTensor = layerInWindowRemainder;

                int layersPerDomainPP = layersInWindow / domainPPSize;
                sharedOutputPPOffset = layerInWindowRemainder / layersPerDomainPP;

                sharedInputBlockId = inputTensorIdx;
                sharedInputLayerId = layerInInputTensor;

                sharedOutputAllLayerOffset = prevOutputLayerOffset + layerInWindowQuotient * layersPerDomainPP
                    + layerInWindowRemainder % layersPerDomainPP;

                break;
            }

            offset += blockNumInWindow * layersInWindow;
            tensorIdxOffset += blockNumInWindow;
            prevOutputLayerOffset += blockNumInWindow * (layersInWindow / domainPPSize);
        }
    }
    __syncthreads();
    inputBlockId = sharedInputBlockId;
    inputLayerId = sharedInputLayerId;
    outputAllLayerOffset = sharedOutputAllLayerOffset;
    outputPPOffset = sharedOutputPPOffset;
}

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitKVCacheForWindowKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int const* blockNumPerWindow, int const* layersPerWindow, int windowNum, int inputAllLayerNum, int tokensPerBlock,
    int headNum, int dimsPerHead, int inputBlockNum, int domainPPSize, int domainTPSize, int headNumDomainTP)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;

    // one thread block per layer

#pragma unroll 1
    for (int threadBlockIdx = blockIdx.x; threadBlockIdx < inputAllLayerNum; threadBlockIdx += gridDim.x)
    {
        int inputBlockId, inputLayerId, outputAllLayerOffset;
        int outputPPOffset;
        getPosId(threadBlockIdx, blockNumPerWindow, layersPerWindow, windowNum, domainPPSize, inputBlockId,
            inputLayerId, outputAllLayerOffset, outputPPOffset);

        T const* inputBlockPtr = inputBlocks[inputBlockId];
        T const* inputLayerPtr = inputBlockPtr + inputLayerId * 2 * headNum * tokensPerBlock * dimsPerHead;
        size_t outputLayerEleOffset = outputAllLayerOffset * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1
        for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
        {
            T const* kInputPtr = inputLayerPtr + headId * tokensPerBlock * dimsPerHead;
            T const* vInputPtr = kInputPtr + headNum * tokensPerBlock * dimsPerHead;
            int outputCacheIdx = headId / headNumDomainTP * domainPPSize + outputPPOffset;
            int headIdInDomainTP = headId % headNumDomainTP;
            T* outputCachePtr = outputCaches[outputCacheIdx];
            T* kOutputPtr = outputCachePtr + outputLayerEleOffset + headIdInDomainTP * tokensPerBlock * dimsPerHead;
            T* vOutputPtr = kOutputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;

#pragma unroll 1
            for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
            {
                auto baseOffset = tokenId * dimsPerHead;
#pragma unroll 1
                for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                     channelId += (subWarpSize * numElePerThread))
                {
                    auto offset = baseOffset + channelId;
                    common::copy<vecSizeByte>(kInputPtr + offset, kOutputPtr + offset);
                    common::copy<vecSizeByte>(vInputPtr + offset, vOutputPtr + offset);
                }
            }
        }
    }
}

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatKVCacheForMLAKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int domainPPSize,
    int domainTPSize, int kvFactor, uint64_t* prefixLayerNumDevPtr)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);
#pragma unroll 1

            for (int headId = 0; headId < headNum; headId++)
            {
                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                int inputCacheIdx = rankInDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];
                int headIdInDomainTP = headId;

                T const* kInputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId)
                        * (static_cast<int64_t>(layerNumInSpecPP * kvFactor * headNum * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;
                int const kvOffset = headNum * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpId; tokenId < tokensPerBlock; tokenId += subWarpNum)
                {
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += subWarpSize * numElePerThread)
                    {
#pragma unroll 1
                        for (int kvId = 0; kvId < kvFactor; kvId++)
                        {
                            common::copy<vecSizeByte>(
                                iKPtr + kvId * kvOffset + channelId, oKPtr + kvId * kvOffset + channelId);
                        }
                    }
                }
            }
        }
    }
}

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void concatKVCacheKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int domainPPSize,
    int DomainTPSize, int headNumDomainTP, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T* vOutputPtr = outputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int inputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];

                int headIdInDomainTP = headId % headNumDomainTP;
                T const* kInputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId)
                        * (static_cast<int64_t>(layerNumInSpecPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;

                T const* vInputPtr = kInputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
                {
                    auto baseOffset = tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        auto offset = baseOffset + channelId;
                        common::copy<vecSizeByte>(kInputPtr + offset, kOutputPtr + offset);
                        common::copy<vecSizeByte>(vInputPtr + offset, vOutputPtr + offset);
                    }
                }
            }
        }
    }
}

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
// __maxnreg__(32)
__global__ void concatKVCacheForWindowKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int const* blockNumPerWindow, int const* layersPerWindow, int windowNum, int outputAllLayerNum, int tokensPerBlock,
    int headNum, int dimsPerHead, int outputBlockNum, int domainPPSize, int domainTPSize, int headNumDomainTP)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup;
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;

#pragma unroll 1
    for (int threadBlockIdx = blockIdx.x; threadBlockIdx < outputAllLayerNum; threadBlockIdx += gridDim.x)
    {
        int outputBlockId, outputLayerId, inputAllLayerOffset;
        int inputPPOffset;
        getPosId(threadBlockIdx, blockNumPerWindow, layersPerWindow, windowNum, domainPPSize, outputBlockId,
            outputLayerId, inputAllLayerOffset, inputPPOffset);
        T* outputBlockPtr = outputBlocks[outputBlockId];
        T* outputLayerPtr = outputBlockPtr + outputLayerId * 2 * headNum * tokensPerBlock * dimsPerHead;
        size_t inputLayerEleOffset = inputAllLayerOffset * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead;

#pragma unroll 1
        for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
        {
            T* kOutputPtr = outputLayerPtr + headId * tokensPerBlock * dimsPerHead;
            T* vOutputPtr = kOutputPtr + headNum * tokensPerBlock * dimsPerHead;
            int inputCacheIdx = headId / headNumDomainTP * domainPPSize + inputPPOffset;
            int headIdInDomainTP = headId % headNumDomainTP;
            T const* inputCachePtr = inputCaches[inputCacheIdx];
            T const* kInputPtr = inputCachePtr + inputLayerEleOffset + headIdInDomainTP * tokensPerBlock * dimsPerHead;
            T const* vInputPtr = kInputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;

#pragma unroll 1
            for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
            {
                auto baseOffset = tokenId * dimsPerHead;
#pragma unroll 1
                for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                     channelId += (subWarpSize * numElePerThread))
                {
                    auto offset = baseOffset + channelId;
                    common::copy<vecSizeByte>(kInputPtr + offset, kOutputPtr + offset);
                    common::copy<vecSizeByte>(vInputPtr + offset, vOutputPtr + offset);
                }
            }
        }
    }
}

template <typename T>
void splitKVCache(std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> const& kVCacheBlocksPerWindow,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager,
    bool isIndexerKCache = false)
{

    size_t inputBlockNumSum = 0;
    for (auto const& [window, blocks] : kVCacheBlocksPerWindow)
    {
        inputBlockNumSum += blocks.size();
    }
    auto targetRankInfo = targetIRanks(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(
            targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize * targetRankInfo.mDomainCPSize)));
    auto outputCacheNum = targetRankInfo.mIRanks.size();
    if (selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        outputCacheNum = targetRankInfo.mDomainPPSize * targetRankInfo.mDomainCPSize;
    }
    else
    {
        outputCacheNum = outputCacheNum / targetRankInfo.mPeerDupHeadFactor;
    }
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNumSum > 0);
    // we want to reduce the call of `cudaMemcpyAsync H2D` , cachePtrs is used to store the pointers of the cache blocks
    // and the values of the prefix layer num.
    std::vector<uint64_t> cachePtrs;
    std::vector<SizeType32> windowSizes;
    std::vector<SizeType32> blockNumInwindow;
    std::vector<SizeType32> layersInWindow;
    size_t cacheBlockSizeSum = 0;
    size_t inputBlockLayerNumSum = 0;
    auto cacheDataType
        = isIndexerKCache ? nvinfer1::DataType::kUINT8 : kVCacheBlocksPerWindow.begin()->second.front()->getDataType();

    for (auto const& [window, blocks] : kVCacheBlocksPerWindow)
    {
        auto cacheBlockSize = blocks.front()->getSize();
        auto cacheDataType = blocks.front()->getDataType();
        windowSizes.push_back(window);
        blockNumInwindow.push_back(blocks.size());
        TLLM_LOG_DEBUG("window: %d, blockNum: %d  blockshape:[%d,%d]", window, blocks.size(),
            blocks.front()->getShape().d[0], blocks.front()->getShape().d[1]);
        auto layersNum = blocks.front()->getDimension<1>();
        layersInWindow.push_back(layersNum);
        for (auto&& kvCacheBlock : blocks)
        {
            TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
            TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
            cacheBlockSizeSum += kvCacheBlock->getSize();
            cachePtrs.push_back(reinterpret_cast<uint64_t>((kvCacheBlock->data())));
            inputBlockLayerNumSum += layersNum;
        }
    }

    for (auto&& outputSplitBlock : outputSplitBlocks)
    {
        TLLM_CHECK_WITH_INFO(outputSplitBlock->getDataType() == cacheDataType,
            "outputSplitBlock data type mismatch expected: %d, actual: %d", static_cast<int>(cacheDataType),
            static_cast<int>(outputSplitBlock->getDataType()));
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputSplitBlock->data()));
    }
    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());
    bool const isWindow = windowSizes.size() > 1;

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(T*));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    runtime::BufferManager::IBufferPtr windowInfoDeviceBuffer;
    std::vector<SizeType32> windowInfoHostBuffer;
    if (isWindow)
    {
        windowInfoHostBuffer.reserve(windowSizes.size() * 2);

        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), blockNumInwindow.begin(), blockNumInwindow.end());
        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), layersInWindow.begin(), layersInWindow.end());
        windowInfoDeviceBuffer = bufferManager.gpu(windowInfoHostBuffer.size(), nvinfer1::DataType::kINT32);
        bufferManager.copy(windowInfoHostBuffer.data(), *windowInfoDeviceBuffer, runtime::MemoryType::kCPU);

        for (auto layerNum : layersInWindow)
        {

            TLLM_CHECK_WITH_INFO(
                layerNum % targetRankInfo.mDomainPPSize == 0, "layerNum in Window must be divisible by domainPPSize");
        }
    }
    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    constexpr int blockDimx = 128;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();
    auto const& destParallelConfig = destCacheState.getParallelConfig();
    auto const& selfAttentionConfig = selfCacheState.getAttentionConfig();

    int oPPNum = selfParallelConfig.mPipelineParallelism;

    // layers
    unsigned int gridDimx = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    // blockNum
    unsigned int gridDimy = inputBlockNumSum;

    int const* blockNumPerWindowDevPtr = nullptr;
    int const* layersPerWindowDevPtr = nullptr;
    int windowNum = windowSizes.size();

    if (isWindow)
    {
        gridDimx = inputBlockLayerNumSum;
        gridDimy = 1;
        blockNumPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data());
        layersPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data()) + windowSizes.size();

        TLLM_LOG_DEBUG("windowNum:%d, inputBlockLayerNumSum:%d, ", windowNum, inputBlockLayerNumSum);
    }

    dim3 gridDim{gridDimx, gridDimy};

    T const** inputBlockPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNumSum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNumSum + outputSplitBlocks.size();

    int const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    int const selfPPRank = selfIdx / (selfParallelConfig.mTensorParallelism * selfParallelConfig.mContextParallelism);
    int const numLayers = selfParallelConfig.mAttentionLayerNumPerPP.at(selfPPRank);
    int const headNum = selfModelConfig.mNbKvHeadsPerLayer[0];

    int const dimsPerHead = computeDimsPerHead(selfCacheState, isIndexerKCache);
    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const domainCPSize = targetRankInfo.mDomainCPSize;
    int const headNumDomainTP = headNum / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);
    int const kvFactor = selfAttentionConfig.mKvFactor;
    bool const isMLA = selfAttentionConfig.mAttentionType == CacheState::AttentionType::kMLA;
    constexpr int mlaSubWarpSize = 16;

    TLLM_LOG_DEBUG(
        "splitKVCache - numLayers: %d, headNum: %d, domainPPSize: %d, domainTPSize: %d, "
        "headsPerDomainTP: %d",
        numLayers, headNum, domainPPSize, domainTPSize, headNumDomainTP);

    int const remainder = dimsPerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernel<T, mlaSubWarpSize, 16><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputBlockPtrsDev, outputCachePtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead,
                inputBlockNumSum, domainPPSize, domainTPSize, domainCPSize, kvFactor, prefixLayerNumDevPtr);
        }
        else if (isWindow)
        {
            splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr, domainCPSize);
        }
        break;
    }
    case 8:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernel<T, mlaSubWarpSize, 8><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputBlockPtrsDev, outputCachePtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead,
                inputBlockNumSum, domainPPSize, domainTPSize, domainCPSize, kvFactor, prefixLayerNumDevPtr);
        }
        else if (isWindow)
        {
            splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr, domainCPSize);
        }
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            if (isMLA)
            {
                splitKVCacheForMLAKernel<T, mlaSubWarpSize, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        domainCPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum,
                        tokensPerBlock, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP, prefixLayerNumDevPtr, domainCPSize);
            }
            break;
        }
    }

    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {
            if (isMLA)
            {
                splitKVCacheForMLAKernel<T, mlaSubWarpSize, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        domainCPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum,
                        tokensPerBlock, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP, prefixLayerNumDevPtr, domainCPSize);
            }
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            if (isMLA)
            {
                splitKVCacheForMLAKernel<T, mlaSubWarpSize, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        domainCPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum,
                        tokensPerBlock, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP, prefixLayerNumDevPtr, domainCPSize);
            }
            break;
        }
        else
        {
            TLLM_THROW("splitKVCacheDispatch encountered an unsupported data type error.");
        }
    }
    }
}

void splitKVCacheDispatch(std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> const& kVCacheBlocksPerWindow,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager,
    bool isIndexerKCache)
{
    TLLM_CHECK(!kVCacheBlocksPerWindow.empty());
    auto dataType
        = isIndexerKCache ? nvinfer1::DataType::kUINT8 : kVCacheBlocksPerWindow.begin()->second.front()->getDataType();

    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        splitKVCache<int64_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 4:
    {
        splitKVCache<int32_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 2:
    {
        splitKVCache<int16_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 1:
    {
        splitKVCache<int8_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    default:
    {
        TLLM_THROW("splitKVCacheDispatch encountered an unsupported data type error.");
    }
    }
}

template <typename T>
void concatKVCache(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>>& outputKvCacheBlocksPerWindow,
    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager, bool isIndexerKCache)
{

    size_t outputBlockNumSum = 0;
    for (auto const& [window, blocks] : outputKvCacheBlocksPerWindow)
    {
        outputBlockNumSum += blocks.size();
    }

    auto targetRankInfo = targetIRanks(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    auto inputCacheNum = targetRankInfo.mIRanks.size();
    if (selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        inputCacheNum = targetRankInfo.mDomainPPSize;
    }
    else
    {
        inputCacheNum = inputCacheNum / targetRankInfo.mPeerDupHeadFactor;
    }
    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());
    TLLM_CHECK(outputBlockNumSum > 0);

    std::vector<uint64_t> cachePtrs;
    std::vector<SizeType32> windowSizes;
    std::vector<SizeType32> blockNumInwindow;
    std::vector<SizeType32> layersInWindow;
    size_t cacheBlockSizeSum = 0;
    size_t outputBlockLayerNumSum = 0;
    auto cacheDataType = inputSplitBlocks.front()->getDataType();

    for (auto const& [window, blocks] : outputKvCacheBlocksPerWindow)
    {
        auto cacheBlockSize = blocks.front()->getSize();
        auto cacheDataType = blocks.front()->getDataType();
        windowSizes.push_back(window);
        blockNumInwindow.push_back(blocks.size());
        auto layersNum = blocks.front()->getDimension<1>();
        layersInWindow.push_back(layersNum);
        for (auto&& kvCacheBlock : blocks)
        {
            TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
            TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
            cachePtrs.push_back(reinterpret_cast<uint64_t>(kvCacheBlock->data()));
            cacheBlockSizeSum += kvCacheBlock->getSize();
        }
        outputBlockLayerNumSum += layersNum * blocks.size();
    }
    for (auto&& inputSplitBlock : inputSplitBlocks)
    {
        TLLM_CHECK(inputSplitBlock->getDataType() == cacheDataType);
        cachePtrs.push_back(reinterpret_cast<uint64_t>(inputSplitBlock->data()));
    }

    // the prefix layer num is used to store the layer num of the previous PP ranks.
    // which is helpful for the kernel to get layer num info.  refer to the function
    // `getLayerIdInDomainPPandRankInDomainPP`.

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);
    bool const isWindow = windowSizes.size() > 1;
    runtime::BufferManager::IBufferPtr windowInfoDeviceBuffer;
    std::vector<SizeType32> windowInfoHostBuffer;
    if (isWindow)
    {
        windowInfoHostBuffer.reserve(windowSizes.size() * 2);

        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), blockNumInwindow.begin(), blockNumInwindow.end());
        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), layersInWindow.begin(), layersInWindow.end());
        windowInfoDeviceBuffer = bufferManager.gpu(windowInfoHostBuffer.size(), nvinfer1::DataType::kINT32);
        bufferManager.copy(windowInfoHostBuffer.data(), *windowInfoDeviceBuffer, runtime::MemoryType::kCPU);
    }
    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();
    auto const& destParallelConfig = destCacheState.getParallelConfig();
    auto const& selfAttentionConfig = selfCacheState.getAttentionConfig();

    int oPPNum = selfParallelConfig.mPipelineParallelism;
    // layers
    unsigned int gridDimx = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    // blockNum
    int const* blockNumPerWindowDevPtr = nullptr;
    int const* layersPerWindowDevPtr = nullptr;
    int windowNum = windowSizes.size();
    unsigned int gridDimy = outputBlockNumSum;
    if (isWindow)
    {
        gridDimx = outputBlockLayerNumSum;
        gridDimy = 1;
        blockNumPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data());
        layersPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data()) + windowSizes.size();
    }
    dim3 gridDim{gridDimx, gridDimy};
    int const endLayerId = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    T** ouptutBlockPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputSplitBlockPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNumSum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + outputBlockNumSum + inputSplitBlocks.size();
    int const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    int const selfPPRank = selfIdx / (selfParallelConfig.mTensorParallelism * selfParallelConfig.mContextParallelism);
    int const numLayers = selfParallelConfig.mAttentionLayerNumPerPP.at(selfPPRank);
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "concatKVCache numLayers:%d", numLayers);
    int const headNum = selfModelConfig.mNbKvHeadsPerLayer[0];
    int const dimsPerHead = computeDimsPerHead(selfCacheState, isIndexerKCache);

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;

    int const headNumDomainTP
        = headNum / (domainTPSize / targetRankInfo.mPeerDupHeadFactor); // TODO: duplicate head factor
    int const kvFactor = selfAttentionConfig.mKvFactor;

    bool isMLA = selfAttentionConfig.mAttentionType == CacheState::AttentionType::kMLA;
    TLLM_LOG_DEBUG(
        "concatKVCache - numLayers: %d, headNum: %d, domainPPSize: %d, domainTPSize: %d, "
        "headsPerDomainTP: %d",
        numLayers, headNum, domainPPSize, domainTPSize, headNumDomainTP);

    int const remainder = dimsPerHead * sizeof(T) % 16;

    int const mlaSubWarpSize = 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            concatKVCacheForMLAKernel<T, mlaSubWarpSize, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize,
                    kvFactor, prefixLayerNumDevPtr);
        }
        else if (isWindow)
        {
            concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, outputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
        }
        break;
    }
    case 8:
    {
        if (isMLA)
        {
            concatKVCacheForMLAKernel<T, mlaSubWarpSize, 8><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputSplitBlockPtrsDev, ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead,
                outputBlockNumSum, domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
        }
        else if (isWindow)
        {
            concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, outputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
        }
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            if (isMLA)
            {
                concatKVCacheForMLAKernel<T, mlaSubWarpSize, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum,
                        outputBlockLayerNumSum, tokensPerBlock, headNum, dimsPerHead, outputBlockNumSum, domainPPSize,
                        domainTPSize, headNumDomainTP);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, headNumDomainTP, prefixLayerNumDevPtr);
            }

            break;
        }
    }
    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {
            if (isMLA)
            {
                concatKVCacheForMLAKernel<T, mlaSubWarpSize, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum,
                        outputBlockLayerNumSum, tokensPerBlock, headNum, dimsPerHead, outputBlockNumSum, domainPPSize,
                        domainTPSize, headNumDomainTP);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, headNumDomainTP, prefixLayerNumDevPtr);
            }
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            if (isMLA)
            {
                concatKVCacheForMLAKernel<T, mlaSubWarpSize, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum,
                        outputBlockLayerNumSum, tokensPerBlock, headNum, dimsPerHead, outputBlockNumSum, domainPPSize,
                        domainTPSize, headNumDomainTP);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, headNumDomainTP, prefixLayerNumDevPtr);
            }
            break;
        }
        else
        {
            TLLM_THROW("concatKVCache encountered an unsupported data type error.");
        }
    }
    }
}

void concatKvCacheV2Dispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>>& outputKvCacheBlocksPerWindow,
    kv_cache::CacheState const& iCacheState, kv_cache::CacheState const& oCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager, bool isIndexerKCache)
{

    auto dataType = outputKvCacheBlocksPerWindow.begin()->second.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        concatKVCache<int64_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 4:
    {
        concatKVCache<int32_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 2:
    {
        concatKVCache<int16_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 1:
    {
        concatKVCache<int8_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    default:
    {
        TLLM_THROW("concatKVCache encountered an unsupported data type error.");
    }
    }
}

} // namespace tensorrt_llm::executor::kv_cache
""",
    'key-files/cpp/tensorrt_llm/executor/serialization.cpp': r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestImpl.h"
#include "tensorrt_llm/executor/responseImpl.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{

// TimePoint
RequestPerfMetrics::TimePoint Serialization::deserializeTimePoint(std::istream& is)
{
    auto serialized_time = su::deserialize<uint64_t>(is);
    std::chrono::microseconds duration(serialized_time);
    return std::chrono::steady_clock::time_point(duration);
}

void Serialization::serialize(RequestPerfMetrics::TimePoint const& tp, std::ostream& os)
{
    su::serialize(std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count(), os);
}

size_t Serialization::serializedSize(RequestPerfMetrics::TimePoint const& /*unused*/)
{
    return sizeof(RequestPerfMetrics::TimePoint);
}

// RequestPerfMetrics
RequestPerfMetrics Serialization::deserializeRequestPerfMetrics(std::istream& is)
{
    auto arrivalTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto firstScheduledTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto firstTokenTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto lastTokenTime = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto kvCacheTransferStart = su::deserialize<RequestPerfMetrics::TimePoint>(is);
    auto kvCacheTransferEnd = su::deserialize<RequestPerfMetrics::TimePoint>(is);

    auto kvCacheSize = su::deserialize<size_t>(is);

    auto numTotalAllocatedBlocks = su::deserialize<SizeType32>(is);
    auto numNewAllocatedBlocks = su::deserialize<SizeType32>(is);
    auto numReusedBlocks = su::deserialize<SizeType32>(is);
    auto numMissedBlocks = su::deserialize<SizeType32>(is);
    auto kvCacheHitRate = su::deserialize<FloatType>(is);

    auto acceptanceRate = su::deserialize<FloatType>(is);
    auto totalAcceptedDraftTokens = su::deserialize<SizeType32>(is);
    auto totalDraftTokens = su::deserialize<SizeType32>(is);

    auto firstIter = su::deserialize<std::optional<IterationType>>(is);
    auto lastIter = su::deserialize<std::optional<IterationType>>(is);
    auto iter = su::deserialize<std::optional<IterationType>>(is);

    RequestPerfMetrics::TimingMetrics timingMetrics{arrivalTime, firstScheduledTime, firstTokenTime, lastTokenTime,
        kvCacheTransferStart, kvCacheTransferEnd, kvCacheSize};
    RequestPerfMetrics::KvCacheMetrics kvCacheMetrics{
        numTotalAllocatedBlocks, numNewAllocatedBlocks, numReusedBlocks, numMissedBlocks, kvCacheHitRate};
    RequestPerfMetrics::SpeculativeDecodingMetrics specDecMetrics{
        acceptanceRate, totalAcceptedDraftTokens, totalDraftTokens};
    return RequestPerfMetrics{timingMetrics, kvCacheMetrics, specDecMetrics, firstIter, lastIter, iter};
}

void Serialization::serialize(RequestPerfMetrics const& metrics, std::ostream& os)
{
    su::serialize(metrics.timingMetrics.arrivalTime, os);
    su::serialize(metrics.timingMetrics.firstScheduledTime, os);
    su::serialize(metrics.timingMetrics.firstTokenTime, os);
    su::serialize(metrics.timingMetrics.lastTokenTime, os);
    su::serialize(metrics.timingMetrics.kvCacheTransferStart, os);
    su::serialize(metrics.timingMetrics.kvCacheTransferEnd, os);

    su::serialize(metrics.timingMetrics.kvCacheSize, os);

    su::serialize(metrics.kvCacheMetrics.numTotalAllocatedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.numNewAllocatedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.numReusedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.numMissedBlocks, os);
    su::serialize(metrics.kvCacheMetrics.kvCacheHitRate, os);

    su::serialize(metrics.speculativeDecoding.acceptanceRate, os);
    su::serialize(metrics.speculativeDecoding.totalAcceptedDraftTokens, os);
    su::serialize(metrics.speculativeDecoding.totalDraftTokens, os);

    su::serialize(metrics.firstIter, os);
    su::serialize(metrics.lastIter, os);
    su::serialize(metrics.iter, os);
}

size_t Serialization::serializedSize(RequestPerfMetrics const& metrics)
{
    size_t totalSize = 0;

    totalSize += su::serializedSize(metrics.timingMetrics.arrivalTime);
    totalSize += su::serializedSize(metrics.timingMetrics.firstScheduledTime);
    totalSize += su::serializedSize(metrics.timingMetrics.firstTokenTime);
    totalSize += su::serializedSize(metrics.timingMetrics.lastTokenTime);
    totalSize += su::serializedSize(metrics.timingMetrics.kvCacheTransferStart);
    totalSize += su::serializedSize(metrics.timingMetrics.kvCacheTransferEnd);

    totalSize += su::serializedSize(metrics.timingMetrics.kvCacheSize);

    totalSize += su::serializedSize(metrics.kvCacheMetrics.numTotalAllocatedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.numNewAllocatedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.numReusedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.numMissedBlocks);
    totalSize += su::serializedSize(metrics.kvCacheMetrics.kvCacheHitRate);

    totalSize += su::serializedSize(metrics.speculativeDecoding.acceptanceRate);
    totalSize += su::serializedSize(metrics.speculativeDecoding.totalAcceptedDraftTokens);
    totalSize += su::serializedSize(metrics.speculativeDecoding.totalDraftTokens);

    totalSize += su::serializedSize(metrics.firstIter);
    totalSize += su::serializedSize(metrics.lastIter);
    totalSize += su::serializedSize(metrics.iter);

    return totalSize;
}

// SamplingConfig
SamplingConfig Serialization::deserializeSamplingConfig(std::istream& is)
{
    auto beamWidth = su::deserialize<SizeType32>(is);
    auto topK = su::deserialize<std::optional<SizeType32>>(is);
    auto topP = su::deserialize<std::optional<FloatType>>(is);
    auto topPMin = su::deserialize<std::optional<FloatType>>(is);
    auto topPResetIds = su::deserialize<std::optional<TokenIdType>>(is);
    auto topPDecay = su::deserialize<std::optional<FloatType>>(is);
    auto randomSeed = su::deserialize<std::optional<RandomSeedType>>(is);
    auto temperature = su::deserialize<std::optional<FloatType>>(is);
    auto minLength = su::deserialize<std::optional<SizeType32>>(is);
    auto beamSearchDiversityRate = su::deserialize<std::optional<FloatType>>(is);
    auto repetitionPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto presencePenalty = su::deserialize<std::optional<FloatType>>(is);
    auto frequencyPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto promptIgnoreLength = su::deserialize<std::optional<SizeType32>>(is);
    auto lengthPenalty = su::deserialize<std::optional<FloatType>>(is);
    auto earlyStopping = su::deserialize<std::optional<SizeType32>>(is);
    auto noRepeatNgramSize = su::deserialize<std::optional<SizeType32>>(is);
    auto numReturnSequences = su::deserialize<std::optional<SizeType32>>(is);
    auto minP = su::deserialize<std::optional<FloatType>>(is);
    auto beamWidthArray = su::deserialize<std::optional<std::vector<SizeType32>>>(is);

    return SamplingConfig{beamWidth, topK, topP, topPMin, topPResetIds, topPDecay, randomSeed, temperature, minLength,
        beamSearchDiversityRate, repetitionPenalty, presencePenalty, frequencyPenalty, promptIgnoreLength,
        lengthPenalty, earlyStopping, noRepeatNgramSize, numReturnSequences, minP, beamWidthArray};
}

void Serialization::serialize(SamplingConfig const& config, std::ostream& os)
{
    su::serialize(config.mBeamWidth, os);
    su::serialize(config.mTopK, os);
    su::serialize(config.mTopP, os);
    su::serialize(config.mTopPMin, os);
    su::serialize(config.mTopPResetIds, os);
    su::serialize(config.mTopPDecay, os);
    su::serialize(config.mSeed, os);
    su::serialize(config.mTemperature, os);
    su::serialize(config.mMinTokens, os);
    su::serialize(config.mBeamSearchDiversityRate, os);
    su::serialize(config.mRepetitionPenalty, os);
    su::serialize(config.mPresencePenalty, os);
    su::serialize(config.mFrequencyPenalty, os);
    su::serialize(config.mPromptIgnoreLength, os);
    su::serialize(config.mLengthPenalty, os);
    su::serialize(config.mEarlyStopping, os);
    su::serialize(config.mNoRepeatNgramSize, os);
    su::serialize(config.mNumReturnSequences, os);
    su::serialize(config.mMinP, os);
    su::serialize(config.mBeamWidthArray, os);
}

size_t Serialization::serializedSize(SamplingConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mBeamWidth);
    totalSize += su::serializedSize(config.mTopK);
    totalSize += su::serializedSize(config.mTopP);
    totalSize += su::serializedSize(config.mTopPMin);
    totalSize += su::serializedSize(config.mTopPResetIds);
    totalSize += su::serializedSize(config.mTopPDecay);
    totalSize += su::serializedSize(config.mSeed);
    totalSize += su::serializedSize(config.mTemperature);
    totalSize += su::serializedSize(config.mMinTokens);
    totalSize += su::serializedSize(config.mBeamSearchDiversityRate);
    totalSize += su::serializedSize(config.mRepetitionPenalty);
    totalSize += su::serializedSize(config.mPresencePenalty);
    totalSize += su::serializedSize(config.mFrequencyPenalty);
    totalSize += su::serializedSize(config.mPromptIgnoreLength);
    totalSize += su::serializedSize(config.mLengthPenalty);
    totalSize += su::serializedSize(config.mEarlyStopping);
    totalSize += su::serializedSize(config.mNoRepeatNgramSize);
    totalSize += su::serializedSize(config.mNumReturnSequences);
    totalSize += su::serializedSize(config.mMinP);
    totalSize += su::serializedSize(config.mBeamWidthArray);
    return totalSize;
}

// OutputConfig
OutputConfig Serialization::deserializeOutputConfig(std::istream& is)
{
    auto returnLogProbs = su::deserialize<bool>(is);
    auto returnContextLogits = su::deserialize<bool>(is);
    auto returnGenerationLogits = su::deserialize<bool>(is);
    auto excludeInputFromOutput = su::deserialize<bool>(is);
    auto returnEncoderOutput = su::deserialize<bool>(is);
    auto returnPerfMetrics = su::deserialize<bool>(is);
    auto additionalOutputs = su::deserialize<std::optional<std::vector<AdditionalModelOutput>>>(is);
    return OutputConfig{returnLogProbs, returnContextLogits, returnGenerationLogits, excludeInputFromOutput,
        returnEncoderOutput, returnPerfMetrics, additionalOutputs};
}

void Serialization::serialize(OutputConfig const& config, std::ostream& os)
{
    su::serialize(config.returnLogProbs, os);
    su::serialize(config.returnContextLogits, os);
    su::serialize(config.returnGenerationLogits, os);
    su::serialize(config.excludeInputFromOutput, os);
    su::serialize(config.returnEncoderOutput, os);
    su::serialize(config.returnPerfMetrics, os);
    su::serialize(config.additionalModelOutputs, os);
}

size_t Serialization::serializedSize(OutputConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.returnLogProbs);
    totalSize += su::serializedSize(config.returnContextLogits);
    totalSize += su::serializedSize(config.returnGenerationLogits);
    totalSize += su::serializedSize(config.excludeInputFromOutput);
    totalSize += su::serializedSize(config.returnEncoderOutput);
    totalSize += su::serializedSize(config.returnPerfMetrics);
    totalSize += su::serializedSize(config.additionalModelOutputs);
    return totalSize;
}

// AdditionalModelOutput
AdditionalModelOutput Serialization::deserializeAdditionalModelOutput(std::istream& is)
{
    auto name = su::deserialize<std::string>(is);
    auto gatherContext = su::deserialize<bool>(is);
    return AdditionalModelOutput{name, gatherContext};
}

void Serialization::serialize(AdditionalModelOutput const& additionalModelOutput, std::ostream& os)
{
    su::serialize(additionalModelOutput.name, os);
    su::serialize(additionalModelOutput.gatherContext, os);
}

size_t Serialization::serializedSize(AdditionalModelOutput const& additionalModelOutput)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(additionalModelOutput.name);
    totalSize += su::serializedSize(additionalModelOutput.gatherContext);
    return totalSize;
}

// ExternalDraftTokensConfig
ExternalDraftTokensConfig Serialization::deserializeExternalDraftTokensConfig(std::istream& is)
{
    auto tokens = su::deserialize<VecTokens>(is);
    auto logits = su::deserialize<std::optional<Tensor>>(is);
    auto acceptanceThreshold = su::deserialize<std::optional<FloatType>>(is);
    auto fastLogits = su::deserialize<std::optional<bool>>(is);

    return ExternalDraftTokensConfig{std::move(tokens), std::move(logits), acceptanceThreshold, fastLogits};
}

void Serialization::serialize(ExternalDraftTokensConfig const& config, std::ostream& os)
{
    su::serialize(config.mTokens, os);
    su::serialize(config.mLogits, os);
    su::serialize(config.mAcceptanceThreshold, os);
    su::serialize(config.mFastLogits, os);
}

size_t Serialization::serializedSize(ExternalDraftTokensConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mTokens);
    totalSize += su::serializedSize(config.mLogits);
    totalSize += su::serializedSize(config.mAcceptanceThreshold);
    totalSize += su::serializedSize(config.mFastLogits);
    return totalSize;
}

// PromptTuningConfig
PromptTuningConfig Serialization::deserializePromptTuningConfig(std::istream& is)
{
    auto tensor = su::deserialize<Tensor>(is);
    auto inputTokenExtraIds = su::deserialize<std::optional<VecTokenExtraIds>>(is);
    return PromptTuningConfig{std::move(tensor), std::move(inputTokenExtraIds)};
}

void Serialization::serialize(PromptTuningConfig const& config, std::ostream& os)
{
    su::serialize(config.mEmbeddingTable, os);
    su::serialize(config.mInputTokenExtraIds, os);
}

size_t Serialization::serializedSize(PromptTuningConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mEmbeddingTable);
    totalSize += su::serializedSize(config.mInputTokenExtraIds);
    return totalSize;
}

// MultimodalInput
MultimodalInput Serialization::deserializeMultimodalInput(std::istream& is)
{
    auto multimodalHashes = su::deserialize<std::vector<std::vector<SizeType32>>>(is);
    auto multimodalPositions = su::deserialize<std::vector<SizeType32>>(is);
    auto multimodalLengths = su::deserialize<std::vector<SizeType32>>(is);
    auto multimodalUuids = su::deserialize<std::optional<std::vector<std::optional<std::string>>>>(is);
    return MultimodalInput{std::move(multimodalHashes), std::move(multimodalPositions), std::move(multimodalLengths),
        std::move(multimodalUuids)};
}

void Serialization::serialize(MultimodalInput const& multimodalInput, std::ostream& os)
{
    su::serialize(multimodalInput.mMultimodalHashes, os);
    su::serialize(multimodalInput.mMultimodalPositions, os);
    su::serialize(multimodalInput.mMultimodalLengths, os);
    su::serialize(multimodalInput.mMultimodalUuids, os);
}

size_t Serialization::serializedSize(MultimodalInput const& multimodalInput)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(multimodalInput.mMultimodalHashes);
    totalSize += su::serializedSize(multimodalInput.mMultimodalPositions);
    totalSize += su::serializedSize(multimodalInput.mMultimodalLengths);
    totalSize += su::serializedSize(multimodalInput.mMultimodalUuids);
    return totalSize;
}

// MropeConfig
MropeConfig Serialization::deserializeMropeConfig(std::istream& is)
{
    auto mropeRotaryCosSin = su::deserialize<Tensor>(is);
    auto mropePositionDeltas = su::deserialize<SizeType32>(is);
    return MropeConfig{std::move(mropeRotaryCosSin), std::move(mropePositionDeltas)};
}

void Serialization::serialize(MropeConfig const& config, std::ostream& os)
{
    su::serialize(config.mMRopeRotaryCosSin, os);
    su::serialize(config.mMRopePositionDeltas, os);
}

size_t Serialization::serializedSize(MropeConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mMRopeRotaryCosSin);
    totalSize += su::serializedSize(config.mMRopePositionDeltas);

    return totalSize;
}

// LoraConfig
LoraConfig Serialization::deserializeLoraConfig(std::istream& is)
{
    auto taskId = su::deserialize<IdType>(is);
    auto weights = su::deserialize<std::optional<Tensor>>(is);
    auto config = su::deserialize<std::optional<Tensor>>(is);
    return LoraConfig{taskId, std::move(weights), std::move(config)};
}

void Serialization::serialize(LoraConfig const& config, std::ostream& os)
{
    su::serialize(config.mTaskId, os);
    su::serialize(config.mWeights, os);
    su::serialize(config.mConfig, os);
}

size_t Serialization::serializedSize(LoraConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.mTaskId);
    totalSize += su::serializedSize(config.mWeights);
    totalSize += su::serializedSize(config.mConfig);
    return totalSize;
}

// SocketState
kv_cache::SocketState Serialization::deserializeSocketState(std::istream& is)
{
    auto port = su::deserialize<decltype(kv_cache::SocketState::mPort)>(is);
    auto ip = su::deserialize<decltype(kv_cache::SocketState::mIp)>(is);
    return kv_cache::SocketState{port, std::move(ip)};
}

void Serialization::serialize(kv_cache::SocketState const& state, std::ostream& os)
{
    su::serialize(state.mPort, os);
    su::serialize(state.mIp, os);
}

size_t Serialization::serializedSize(kv_cache::SocketState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mPort);
    totalSize += su::serializedSize(state.mIp);
    return totalSize;
}

// AgentState
kv_cache::AgentState Serialization::deserializeAgentState(std::istream& is)
{
    auto agentName = su::deserialize<decltype(kv_cache::AgentState::mAgentName)>(is);
    auto connectionInfo = su::deserialize<decltype(kv_cache::AgentState::mConnectionInfo)>(is);
    return kv_cache::AgentState{std::move(agentName), std::move(connectionInfo)};
}

void Serialization::serialize(kv_cache::AgentState const& state, std::ostream& os)
{
    su::serialize(state.mAgentName, os);
    su::serialize(state.mConnectionInfo, os);
}

size_t Serialization::serializedSize(kv_cache::AgentState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mAgentName);
    totalSize += su::serializedSize(state.mConnectionInfo);
    return totalSize;
}

// CommState
kv_cache::CommState Serialization::deserializeCommState(std::istream& is)
{
    auto selfIdx = su::deserialize<decltype(kv_cache::CommState::mSelfIdx)>(is);
    auto variantIdx = su::deserialize<std::size_t>(is);
    constexpr std::size_t mpiIdx{1};
    constexpr std::size_t socketIdx{2};
    constexpr std::size_t agentIdx{3};
    static_assert(
        std::is_same_v<kv_cache::MpiState, std::variant_alternative_t<mpiIdx, decltype(kv_cache::CommState::mState)>>);
    static_assert(std::is_same_v<std::vector<kv_cache::SocketState>,
        std::variant_alternative_t<socketIdx, decltype(kv_cache::CommState::mState)>>);
    if (variantIdx == mpiIdx)
    {
        auto ranks = su::deserialize<decltype(kv_cache::MpiState::mRanks)>(is);
        return kv_cache::CommState{std::move(ranks), selfIdx};
    }
    if (variantIdx == socketIdx)
    {
        auto state = su::deserialize<std::vector<kv_cache::SocketState>>(is);
        return kv_cache::CommState{std::move(state), selfIdx};
    }
    if (variantIdx == agentIdx)
    {
        auto state = su::deserialize<std::vector<kv_cache::AgentState>>(is);
        return kv_cache::CommState{std::move(state), selfIdx};
    }
    return {};
}

void Serialization::serialize(kv_cache::CommState const& state, std::ostream& os)
{
    su::serialize(state.mSelfIdx, os);
    su::serialize(state.mState.index(), os);
    if (state.isMpiState())
    {
        su::serialize(state.getMpiState().mRanks, os);
    }
    else if (state.isSocketState())
    {
        su::serialize(state.getSocketState(), os);
    }
    else if (state.isAgentState())
    {
        su::serialize(state.getAgentState(), os);
    }
    else
    {
        TLLM_THROW("Unknown context phase state communication type.");
    }
}

size_t Serialization::serializedSize(kv_cache::CommState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mSelfIdx);
    totalSize += su::serializedSize(state.mState.index());
    if (state.isMpiState())
    {
        totalSize += su::serializedSize(state.getMpiState().mRanks);
    }
    else if (state.isSocketState())
    {
        totalSize += su::serializedSize(state.getSocketState());
    }
    else if (state.isAgentState())
    {
        totalSize += su::serializedSize(state.getAgentState());
    }
    else
    {
        TLLM_THROW("Unknown context phase state communication type.");
    }
    return totalSize;
}

// CacheState
kv_cache::CacheState Serialization::deserializeCacheState(std::istream& is)
{
    using CacheState = kv_cache::CacheState;
    auto nbKvHeadsPerLayer = su::deserialize<decltype(CacheState::ModelConfig::mNbKvHeadsPerLayer)>(is);
    auto sizePerHead = su::deserialize<decltype(CacheState::ModelConfig::mSizePerHead)>(is);
    auto tokensPerBlock = su::deserialize<decltype(CacheState::ModelConfig::mTokensPerBlock)>(is);
    auto tensorParallelism = su::deserialize<decltype(CacheState::ParallelConfig::mTensorParallelism)>(is);
    auto pipelineParallelism = su::deserialize<decltype(CacheState::ParallelConfig::mPipelineParallelism)>(is);
    auto contextParallelism = su::deserialize<decltype(CacheState::ParallelConfig::mContextParallelism)>(is);
    auto enableAttentionDP = su::deserialize<decltype(CacheState::ParallelConfig::mEnableAttentionDP)>(is);
    auto DPrank = su::deserialize<decltype(CacheState::ParallelConfig::mDPrank)>(is);
    auto DPsize = su::deserialize<decltype(CacheState::ParallelConfig::mDPsize)>(is);
    auto attentionLayerNumPerPP = su::deserialize<decltype(CacheState::ParallelConfig::mAttentionLayerNumPerPP)>(is);
    auto dataType = su::deserialize<decltype(CacheState::mDataType)>(is);
    auto attentionType = su::deserialize<decltype(CacheState::AttentionConfig::mAttentionType)>(is);
    auto kvFactor = su::deserialize<decltype(CacheState::AttentionConfig::mKvFactor)>(is);
    auto enableBlockReuse = su::deserialize<bool>(is);
    auto enablePartialReuse = su::deserialize<bool>(is);
    auto hasIndexerKCache = su::deserialize<bool>(is);
    auto indexerDimPerHead = su::deserialize<decltype(CacheState::ModelConfig::mSizePerHead)>(is);
    auto indexerKCacheQuantBlockSize = su::deserialize<decltype(CacheState::ModelConfig::mTokensPerBlock)>(is);
    auto indexerKCacheUseFp4 = su::deserialize<bool>(is);
    // RNN config (optional)
    auto hasRnnConfig = su::deserialize<bool>(is);
    std::optional<CacheState::RnnModelConfig> rnnModelConfig;
    std::vector<SizeType32> rnnLayerNumPerPP;
    nvinfer1::DataType convStateDataType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType ssmStateDataType{nvinfer1::DataType::kFLOAT};
    if (hasRnnConfig)
    {
        CacheState::RnnModelConfig rnnCfg;
        rnnCfg.mDState = su::deserialize<decltype(CacheState::RnnModelConfig::mDState)>(is);
        rnnCfg.mDConv = su::deserialize<decltype(CacheState::RnnModelConfig::mDConv)>(is);
        rnnCfg.mHiddenSize = su::deserialize<decltype(CacheState::RnnModelConfig::mHiddenSize)>(is);
        rnnCfg.mHeadDim = su::deserialize<decltype(CacheState::RnnModelConfig::mHeadDim)>(is);
        rnnCfg.mConvDimSize = su::deserialize<decltype(CacheState::RnnModelConfig::mConvDimSize)>(is);
        rnnCfg.mNGroups = su::deserialize<decltype(CacheState::RnnModelConfig::mNGroups)>(is);
        rnnCfg.mNumLayers = su::deserialize<decltype(CacheState::RnnModelConfig::mNumLayers)>(is);
        rnnCfg.mNumHeads = su::deserialize<decltype(CacheState::RnnModelConfig::mNumHeads)>(is);
        convStateDataType = su::deserialize<nvinfer1::DataType>(is);
        ssmStateDataType = su::deserialize<nvinfer1::DataType>(is);
        rnnLayerNumPerPP = su::deserialize<std::vector<SizeType32>>(is);
        rnnModelConfig = std::move(rnnCfg);
    }
    CacheState cacheState{nbKvHeadsPerLayer, sizePerHead, tokensPerBlock, tensorParallelism, pipelineParallelism,
        contextParallelism, attentionLayerNumPerPP, dataType, attentionType, kvFactor, enableAttentionDP, DPrank,
        DPsize, enableBlockReuse, enablePartialReuse, hasIndexerKCache, indexerDimPerHead, indexerKCacheQuantBlockSize,
        indexerKCacheUseFp4};
    if (rnnModelConfig.has_value())
    {
        cacheState.setRnnConfig(
            std::move(rnnModelConfig.value()), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
    }
    return cacheState;
}

void Serialization::serialize(kv_cache::CacheState const& state, std::ostream& os)
{
    su::serialize(state.mModelConfig.mNbKvHeadsPerLayer, os);
    su::serialize(state.mModelConfig.mSizePerHead, os);
    su::serialize(state.mModelConfig.mTokensPerBlock, os);
    su::serialize(state.mParallelConfig.mTensorParallelism, os);
    su::serialize(state.mParallelConfig.mPipelineParallelism, os);
    su::serialize(state.mParallelConfig.mContextParallelism, os);
    su::serialize(state.mParallelConfig.mEnableAttentionDP, os);
    su::serialize(state.mParallelConfig.mDPrank, os);
    su::serialize(state.mParallelConfig.mDPsize, os);
    su::serialize(state.mParallelConfig.mAttentionLayerNumPerPP, os);
    su::serialize(state.mDataType, os);
    su::serialize(state.mAttentionConfig.mAttentionType, os);
    su::serialize(state.mAttentionConfig.mKvFactor, os);
    su::serialize(state.mEnableBlockReuse, os);
    su::serialize(state.mEnablePartialReuse, os);
    su::serialize(state.getHasIndexerKCache(), os);
    su::serialize(state.getIndexerDimPerHead(), os);
    su::serialize(state.getIndexerKCacheQuantBlockSize(), os);
    su::serialize(state.getIndexerKCacheUseFp4(), os);
    // RNN config (optional)
    su::serialize(state.mRnnCacheState.has_value(), os);
    if (state.mRnnCacheState.has_value())
    {
        auto const& rnn = state.mRnnCacheState->mModelConfig;
        su::serialize(rnn.mDState, os);
        su::serialize(rnn.mDConv, os);
        su::serialize(rnn.mHiddenSize, os);
        su::serialize(rnn.mHeadDim, os);
        su::serialize(rnn.mConvDimSize, os);
        su::serialize(rnn.mNGroups, os);
        su::serialize(rnn.mNumLayers, os);
        su::serialize(rnn.mNumHeads, os);
        su::serialize(state.mRnnCacheState->mConvStateDataType, os);
        su::serialize(state.mRnnCacheState->mSsmStateDataType, os);
        su::serialize(state.mRnnCacheState->mLayerNumPerPP, os);
    }
}

size_t Serialization::serializedSize(kv_cache::CacheState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mModelConfig.mNbKvHeadsPerLayer);
    totalSize += su::serializedSize(state.mModelConfig.mSizePerHead);
    totalSize += su::serializedSize(state.mModelConfig.mTokensPerBlock);
    totalSize += su::serializedSize(state.mParallelConfig.mTensorParallelism);
    totalSize += su::serializedSize(state.mParallelConfig.mPipelineParallelism);
    totalSize += su::serializedSize(state.mParallelConfig.mContextParallelism);
    totalSize += su::serializedSize(state.mParallelConfig.mEnableAttentionDP);
    totalSize += su::serializedSize(state.mParallelConfig.mDPrank);
    totalSize += su::serializedSize(state.mParallelConfig.mDPsize);
    totalSize += su::serializedSize(state.mParallelConfig.mAttentionLayerNumPerPP);
    totalSize += su::serializedSize(state.mDataType);
    totalSize += su::serializedSize(state.mAttentionConfig.mAttentionType);
    totalSize += su::serializedSize(state.mAttentionConfig.mKvFactor);
    totalSize += su::serializedSize(state.mEnableBlockReuse);
    totalSize += su::serializedSize(state.mEnablePartialReuse);
    totalSize += su::serializedSize(state.getHasIndexerKCache());
    totalSize += su::serializedSize(state.getIndexerDimPerHead());
    totalSize += su::serializedSize(state.getIndexerKCacheQuantBlockSize());
    totalSize += su::serializedSize(state.getIndexerKCacheUseFp4());
    // RNN config (optional)
    totalSize += su::serializedSize(state.mRnnCacheState.has_value());
    if (state.mRnnCacheState.has_value())
    {
        auto const& rnn = state.mRnnCacheState->mModelConfig;
        totalSize += su::serializedSize(rnn.mDState);
        totalSize += su::serializedSize(rnn.mDConv);
        totalSize += su::serializedSize(rnn.mHiddenSize);
        totalSize += su::serializedSize(rnn.mHeadDim);
        totalSize += su::serializedSize(rnn.mConvDimSize);
        totalSize += su::serializedSize(rnn.mNGroups);
        totalSize += su::serializedSize(rnn.mNumLayers);
        totalSize += su::serializedSize(rnn.mNumHeads);
        totalSize += su::serializedSize(state.mRnnCacheState->mConvStateDataType);
        totalSize += su::serializedSize(state.mRnnCacheState->mSsmStateDataType);
        totalSize += su::serializedSize(state.mRnnCacheState->mLayerNumPerPP);
    }
    return totalSize;
}

// DataTransceiverState

DataTransceiverState Serialization::deserializeDataTransceiverState(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    return deserializeDataTransceiverState(is);
}

DataTransceiverState Serialization::deserializeDataTransceiverState(std::istream& is)
{
    DataTransceiverState state;
    auto commState = su::deserialize<decltype(DataTransceiverState::mCommState)>(is);
    if (commState)
    {
        state.setCommState(std::move(commState).value());
    }
    auto cacheState = su::deserialize<decltype(DataTransceiverState::mCacheState)>(is);
    if (cacheState)
    {
        state.setCacheState(std::move(cacheState).value());
    }
    return state;
}

void Serialization::serialize(DataTransceiverState const& state, std::ostream& os)
{
    su::serialize(state.mCommState, os);
    su::serialize(state.mCacheState, os);
}

std::vector<char> Serialization::serialize(DataTransceiverState const& state)
{
    std::vector<char> buffer(serializedSize(state));

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);
    serialize(state, os);
    return buffer;
}

size_t Serialization::serializedSize(DataTransceiverState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mCommState);
    totalSize += su::serializedSize(state.mCacheState);
    return totalSize;
}

// ContextPhaseParams
ContextPhaseParams Serialization::deserializeContextPhaseParams(std::istream& is)
{
    auto reqId = su::deserialize<decltype(ContextPhaseParams::mReqId)>(is);
    auto firstGenTokens = su::deserialize<decltype(ContextPhaseParams::mFirstGenTokens)>(is);
    auto draftTokens = su::deserialize<decltype(ContextPhaseParams::mDraftTokens)>(is);
    auto ctxDpRank = su::deserialize<decltype(ContextPhaseParams::mCtxDpRank)>(is);
    auto disaggInfoEndpoint = su::deserialize<decltype(ContextPhaseParams::mDisaggInfoEndpoint)>(is);
    auto hasState = su::deserialize<bool>(is);
    if (hasState)
    {
        auto state = std::make_unique<DataTransceiverState>();
        *state = deserializeDataTransceiverState(is);
        return ContextPhaseParams{std::move(firstGenTokens), reqId, state.release(), std::move(draftTokens), ctxDpRank,
            std::move(disaggInfoEndpoint)};
    }
    return ContextPhaseParams{
        std::move(firstGenTokens), reqId, std::move(draftTokens), ctxDpRank, std::move(disaggInfoEndpoint)};
}

void Serialization::serialize(ContextPhaseParams const& contextPhaseParams, std::ostream& os)
{
    su::serialize(contextPhaseParams.mReqId, os);
    su::serialize(contextPhaseParams.mFirstGenTokens, os);
    su::serialize(contextPhaseParams.mDraftTokens, os);
    su::serialize(contextPhaseParams.mCtxDpRank, os);
    su::serialize(contextPhaseParams.mDisaggInfoEndpoint, os);
    su::serialize(static_cast<bool>(contextPhaseParams.mState), os);
    if (contextPhaseParams.mState)
    {
        serialize(*static_cast<DataTransceiverState*>(contextPhaseParams.mState.get()), os);
    }
}

size_t Serialization::serializedSize(ContextPhaseParams const& contextPhaseParams)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(contextPhaseParams.mReqId);
    totalSize += su::serializedSize(contextPhaseParams.mFirstGenTokens);
    totalSize += su::serializedSize(contextPhaseParams.mDraftTokens);
    totalSize += su::serializedSize(contextPhaseParams.mCtxDpRank);
    totalSize += su::serializedSize(contextPhaseParams.mDisaggInfoEndpoint);
    totalSize += su::serializedSize(bool{});
    if (contextPhaseParams.mState)
    {
        totalSize += serializedSize(*static_cast<DataTransceiverState*>(contextPhaseParams.mState.get()));
    }
    return totalSize;
}

// Request
Request Serialization::deserializeRequest(std::istream& is)
{
    // Serialization of Request with logitsPostProcessor is currently not supported.
    // Dynamic logitsPostProcessor only supported with replicate=false or no tensor parallelism.
    auto inputTokenIds = su::deserialize<VecTokens>(is);
    auto maxNewTokens = su::deserialize<SizeType32>(is);
    auto streaming = su::deserialize<bool>(is);
    auto samplingConfig = su::deserialize<SamplingConfig>(is);
    auto outputConfig = su::deserialize<OutputConfig>(is);
    auto endId = su::deserialize<std::optional<SizeType32>>(is);
    auto padId = su::deserialize<std::optional<SizeType32>>(is);
    auto positionIds = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto badWords = su::deserialize<std::optional<std::list<VecTokens>>>(is);
    auto stopWords = su::deserialize<std::optional<std::list<VecTokens>>>(is);
    auto embeddingBias = su::deserialize<std::optional<Tensor>>(is);
    auto externalDraftTokensConfig = su::deserialize<std::optional<ExternalDraftTokensConfig>>(is);
    auto pTuningConfig = su::deserialize<std::optional<PromptTuningConfig>>(is);
    auto multimodalInput = su::deserialize<std::optional<MultimodalInput>>(is);
    auto multimodalEmbedding = su::deserialize<std::optional<Tensor>>(is);
    auto mRopeConfig = su::deserialize<std::optional<MropeConfig>>(is);
    auto loraConfig = su::deserialize<std::optional<LoraConfig>>(is);
    auto lookaheadConfig = su::deserialize<std::optional<LookaheadDecodingConfig>>(is);
    auto kvCacheRetentionConfig = su::deserialize<std::optional<KvCacheRetentionConfig>>(is);
    auto logitsPostProcessorName = su::deserialize<std::optional<std::string>>(is);
    auto encoderInputTokenIds = su::deserialize<std::optional<VecTokens>>(is);
    auto clientId = su::deserialize<std::optional<IdType>>(is);
    auto returnAllGeneratedTokens = su::deserialize<bool>(is);
    auto priority = su::deserialize<executor::PriorityType>(is);
    auto requestType = su::deserialize<executor::RequestType>(is);
    auto contextPhaseParams = su::deserialize<std::optional<ContextPhaseParams>>(is);
    auto encoderInputFeatures = su::deserialize<std::optional<Tensor>>(is);
    auto encoderOutputLength = su::deserialize<std::optional<SizeType32>>(is);
    auto crossAttentionMask = su::deserialize<std::optional<Tensor>>(is);
    auto numReturnSequences = su::deserialize<SizeType32>(is);
    auto eagleConfig = su::deserialize<std::optional<EagleConfig>>(is);
    auto skipCrossAttnBlocks = su::deserialize<std::optional<Tensor>>(is);
    auto guidedDecodingParams = su::deserialize<std::optional<GuidedDecodingParams>>(is);
    auto languageAdapterUid = su::deserialize<std::optional<SizeType32>>(is);
    auto allottedTimeInt = su::deserialize<std::optional<std::chrono::milliseconds::rep>>(is);
    auto allottedTimeMs = allottedTimeInt
        ? std::optional<std::chrono::milliseconds>(std::chrono::milliseconds(*allottedTimeInt))
        : std::nullopt;
    auto cacheSaltID = su::deserialize<std::optional<CacheSaltIDType>>(is);
    auto disaggRequestId = su::deserialize<std::optional<IdType>>(is);

    return Request(std::move(inputTokenIds), maxNewTokens, streaming, samplingConfig, outputConfig, endId, padId,
        std::move(positionIds), std::move(badWords), std::move(stopWords), std::move(embeddingBias),
        std::move(externalDraftTokensConfig), std::move(pTuningConfig), std::move(multimodalInput),
        std::move(multimodalEmbedding), std::move(mRopeConfig), std::move(loraConfig), lookaheadConfig,
        std::move(kvCacheRetentionConfig), std::move(logitsPostProcessorName), std::nullopt,
        std::move(encoderInputTokenIds), clientId, returnAllGeneratedTokens, priority, requestType,
        std::move(contextPhaseParams), std::move(encoderInputFeatures), encoderOutputLength,
        std::move(crossAttentionMask), numReturnSequences, std::move(eagleConfig), std::move(skipCrossAttnBlocks),
        std::move(guidedDecodingParams), languageAdapterUid, allottedTimeMs, cacheSaltID, disaggRequestId);
}

void Serialization::serialize(Request const& request, std::ostream& os)
{
    request.mImpl->serialize(os);
}

size_t Serialization::serializedSize(Request const& request)
{
    return request.mImpl->serializedSize();
}

// Tensor
Tensor Serialization::deserializeTensor(std::istream& is)
{
    // DataType
    DataType dataType{};
    is.read(reinterpret_cast<char*>(&dataType), sizeof(dataType));

    // Shape
    size_t shapeSize{0};
    is.read(reinterpret_cast<char*>(&shapeSize), sizeof(size_t));
    static constexpr int32_t MAX_DIMS{8};
    TLLM_CHECK(shapeSize < MAX_DIMS);

    Shape::DimType64 dims[MAX_DIMS];
    is.read(reinterpret_cast<char*>(&dims[0]), shapeSize * sizeof(Shape::DimType64));
    Shape shape(&dims[0], shapeSize);

    // Memory Type
    MemoryType memoryType{};
    is.read(reinterpret_cast<char*>(&memoryType), sizeof(memoryType));

    // Size in bytes
    size_t sizeInBytes{0};
    is.read(reinterpret_cast<char*>(&sizeInBytes), sizeof(size_t));

    Tensor tensor;
    switch (memoryType)
    {
    case MemoryType::kCPU:
    {
        tensor = Tensor::cpu(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kCPU_PINNED:
    {
        tensor = Tensor::pinned(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kUVM:
    {
        tensor = Tensor::managed(dataType, shape);
        is.read(reinterpret_cast<char*>(tensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        break;
    }
    case MemoryType::kGPU:
    {
        // TODO: Eventually we might want to support serialization/deserialization in GPU memory
        //       Until then created Pinned tensor and move to GPU
        auto pinnedTensor = Tensor::pinned(dataType, shape);
        is.read(reinterpret_cast<char*>(pinnedTensor.getData()), static_cast<std::streamsize>(sizeInBytes));
        auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        tensor = pinnedTensor.copyToGpu(stream);
        stream->synchronize();
        break;
    }
    case MemoryType::kUNKNOWN:
    {
        TLLM_THROW("Cannot deserialize tensor with UNKNOWN type.");
        break;
    }
    default:
    {
        TLLM_THROW("Memory type not supported in deserializeTensor.");
        break;
    }
    }

    return tensor;
}

void Serialization::serialize(Tensor const& tensor, std::ostream& os)
{
    auto dataType = tensor.getDataType();
    os.write(reinterpret_cast<char const*>(&dataType), sizeof(dataType));
    auto shape = tensor.getShape();
    auto shapeSize = shape.size();
    os.write(reinterpret_cast<char const*>(&shapeSize), sizeof(shapeSize));
    os.write(reinterpret_cast<char const*>(&shape[0]), shapeSize * sizeof(Shape::DimType64));

    // Memory Type
    auto memoryType = tensor.getMemoryType();
    os.write(reinterpret_cast<char const*>(&memoryType), sizeof(memoryType));

    std::size_t sizeInBytes = tensor.getSizeInBytes();
    os.write(reinterpret_cast<char const*>(&sizeInBytes), sizeof(sizeInBytes));

    if (memoryType == MemoryType::kCPU || memoryType == MemoryType::kCPU_PINNED || memoryType == MemoryType::kUVM)
    {
        void const* data = tensor.getData();
        os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
    }
    // Need special treatment for GPU type
    else if (memoryType == MemoryType::kGPU)
    {
        auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        auto pinnedTensor = tensor.copyToPinned(stream);
        stream->synchronize();
        void const* data = pinnedTensor.getData();
        os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
    }
    else if (memoryType == MemoryType::kUNKNOWN)
    {
        TLLM_THROW("Memory type unknown when serializing tensor");
    }
}

size_t Serialization::serializedSize(Tensor const& tensor)
{
    size_t totalSize = 0;
    totalSize += sizeof(tensor.getDataType()); // datatype
    auto const shape = tensor.getShape();
    auto const shapeSize = shape.size();
    totalSize += sizeof(decltype(shapeSize)); // number of dims
    TLLM_CHECK(shapeSize > 0);
    totalSize += shapeSize * sizeof(decltype(shape[0]));

    auto memoryType = tensor.getMemoryType();
    totalSize += sizeof(memoryType); // memory type

    totalSize += sizeof(size_t);     // Size in bytes
    totalSize += tensor.getSizeInBytes();
    return totalSize;
}

// SpeculativeDecodingFastLogitsInfo
SpeculativeDecodingFastLogitsInfo Serialization::deserializeSpecDecFastLogitsInfo(std::istream& is)
{
    auto draftRequestId = su::deserialize<uint64_t>(is);
    auto draftParticipantId = su::deserialize<int32_t>(is);

    return SpeculativeDecodingFastLogitsInfo{draftRequestId, draftParticipantId};
}

void Serialization::serialize(SpeculativeDecodingFastLogitsInfo const& info, std::ostream& os)
{
    su::serialize(info.draftRequestId, os);
    su::serialize(info.draftParticipantId, os);
}

size_t Serialization::serializedSize(SpeculativeDecodingFastLogitsInfo const& info)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(info.draftRequestId);
    totalSize += su::serializedSize(info.draftParticipantId);
    return totalSize;
}

// Result
Result Serialization::deserializeResult(std::istream& is)
{
    Result result{};
    result.isFinal = su::deserialize<bool>(is);
    result.outputTokenIds = su::deserialize<BeamTokens>(is);
    result.cumLogProbs = su::deserialize<std::optional<VecLogProbs>>(is);
    result.logProbs = su::deserialize<std::optional<std::vector<VecLogProbs>>>(is);
    result.contextLogits = su::deserialize<std::optional<Tensor>>(is);
    result.generationLogits = su::deserialize<std::optional<Tensor>>(is);
    result.specDecFastLogitsInfo = su::deserialize<std::optional<SpeculativeDecodingFastLogitsInfo>>(is);
    result.encoderOutput = su::deserialize<std::optional<Tensor>>(is);
    result.finishReasons = su::deserialize<std::vector<FinishReason>>(is);
    result.contextPhaseParams = su::deserialize<std::optional<ContextPhaseParams>>(is);
    result.decodingIter = su::deserialize<SizeType32>(is);
    result.avgDecodedTokensPerIter = su::deserialize<float>(is);
    result.sequenceIndex = su::deserialize<SizeType32>(is);
    result.isSequenceFinal = su::deserialize<bool>(is);
    result.requestPerfMetrics = su::deserialize<std::optional<RequestPerfMetrics>>(is);
    result.additionalOutputs = su::deserialize<std::vector<AdditionalOutput>>(is);
    return result;
}

void Serialization::serialize(Result const& result, std::ostream& os)
{
    su::serialize(result.isFinal, os);
    su::serialize(result.outputTokenIds, os);
    su::serialize(result.cumLogProbs, os);
    su::serialize(result.logProbs, os);
    su::serialize(result.contextLogits, os);
    su::serialize(result.generationLogits, os);
    su::serialize(result.specDecFastLogitsInfo, os);
    su::serialize(result.encoderOutput, os);
    su::serialize(result.finishReasons, os);
    su::serialize(result.contextPhaseParams, os);
    su::serialize(result.decodingIter, os);
    su::serialize(result.avgDecodedTokensPerIter, os);
    su::serialize(result.sequenceIndex, os);
    su::serialize(result.isSequenceFinal, os);
    su::serialize(result.requestPerfMetrics, os);
    su::serialize(result.additionalOutputs, os);
}

size_t Serialization::serializedSize(Result const& result)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(result.isFinal);
    totalSize += su::serializedSize(result.outputTokenIds);
    totalSize += su::serializedSize(result.cumLogProbs);
    totalSize += su::serializedSize(result.logProbs);
    totalSize += su::serializedSize(result.contextLogits);
    totalSize += su::serializedSize(result.specDecFastLogitsInfo);
    totalSize += su::serializedSize(result.generationLogits);
    totalSize += su::serializedSize(result.encoderOutput);
    totalSize += su::serializedSize(result.finishReasons);
    totalSize += su::serializedSize(result.contextPhaseParams);
    totalSize += su::serializedSize(result.decodingIter);
    totalSize += su::serializedSize(result.avgDecodedTokensPerIter);
    totalSize += su::serializedSize(result.sequenceIndex);
    totalSize += su::serializedSize(result.isSequenceFinal);
    totalSize += su::serializedSize(result.requestPerfMetrics);
    totalSize += su::serializedSize(result.additionalOutputs);
    return totalSize;
}

// Response
Response Serialization::deserializeResponse(std::istream& is)
{
    auto requestId = su::deserialize<IdType>(is);
    auto errOrResult = su::deserialize<std::variant<std::string, Result>>(is);
    auto clientId = su::deserialize<std::optional<IdType>>(is);

    return std::holds_alternative<std::string>(errOrResult)
        ? Response{requestId, std::get<std::string>(errOrResult), clientId}
        : Response{requestId, std::get<Result>(errOrResult), clientId};
}

void Serialization::serialize(Response const& response, std::ostream& os)
{
    su::serialize(response.mImpl->mRequestId, os);
    su::serialize(response.mImpl->mErrOrResult, os);
    su::serialize(response.mImpl->mClientId, os);
}

size_t Serialization::serializedSize(Response const& response)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(response.mImpl->mRequestId);
    totalSize += su::serializedSize(response.mImpl->mErrOrResult);
    totalSize += su::serializedSize(response.mImpl->mClientId);
    return totalSize;
}

// Vector of responses
std::vector<Response> Serialization::deserializeResponses(std::vector<char>& buffer)
{
    std::vector<Response> responses;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numResponses = su::deserialize<std::size_t>(is);
    for (std::size_t resp = 0; resp < numResponses; ++resp)
    {
        responses.emplace_back(Serialization::deserializeResponse(is));
    }
    return responses;
}

std::vector<char> Serialization::serialize(std::vector<Response> const& responses)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& response : responses)
    {
        totalSize += su::serializedSize(response);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    su::serialize(responses.size(), os);
    for (auto const& response : responses)
    {
        su::serialize(response, os);
    }
    return buffer;
}

AdditionalOutput Serialization::deserializeAdditionalOutput(std::istream& is)
{
    return {
        Serialization::deserializeString(is),
        Serialization::deserializeTensor(is),
    };
}

void Serialization::serialize(AdditionalOutput const& additionalOutput, std::ostream& os)
{
    su::serialize(additionalOutput.name, os);
    su::serialize(additionalOutput.output, os);
}

size_t serializedSize(AdditionalOutput const& additionalOutput)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(additionalOutput.name);
    totalSize += su::serializedSize(additionalOutput.output);
    return totalSize;
}

// ExecutorConfig
ExecutorConfig Serialization::deserializeExecutorConfig(std::istream& is)
{
    auto maxBeamWidth = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxBeamWidth)>(is);
    auto maxBatchSize = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxBatchSize)>(is);
    auto maxNumTokens = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxNumTokens)>(is);
    auto schedulerConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getSchedulerConfig)>(is);
    auto kvCacheConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getKvCacheConfig)>(is);
    auto enableChunkedContext = su::deserializeWithGetterType<decltype(&ExecutorConfig::getEnableChunkedContext)>(is);
    auto normalizeLogProbs = su::deserializeWithGetterType<decltype(&ExecutorConfig::getNormalizeLogProbs)>(is);
    auto iterStatsMaxIterations
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getIterStatsMaxIterations)>(is);
    auto requestStatsMaxIterations
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getRequestStatsMaxIterations)>(is);
    auto batchingType = su::deserializeWithGetterType<decltype(&ExecutorConfig::getBatchingType)>(is);
    auto parallelConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getParallelConfig)>(is);
    auto peftCacheConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getPeftCacheConfig)>(is);
    auto decodingConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getDecodingConfig)>(is);
    auto useGpuDirectStorage = su::deserializeWithGetterType<decltype(&ExecutorConfig::getUseGpuDirectStorage)>(is);
    auto gpuWeightsPercent = su::deserializeWithGetterType<decltype(&ExecutorConfig::getGpuWeightsPercent)>(is);
    auto maxQueueSize = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxQueueSize)>(is);
    auto extendedRuntimePerfKnobConfig
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getExtendedRuntimePerfKnobConfig)>(is);
    auto debugConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getDebugConfig)>(is);
    auto recvPollPeriodMs = su::deserializeWithGetterType<decltype(&ExecutorConfig::getRecvPollPeriodMs)>(is);
    auto maxSeqIdleMicroseconds
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getMaxSeqIdleMicroseconds)>(is);
    auto specDecConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getSpecDecConfig)>(is);
    auto guidedDecodingConfig = su::deserializeWithGetterType<decltype(&ExecutorConfig::getGuidedDecodingConfig)>(is);
    auto additionalModelOutputs
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getAdditionalModelOutputs)>(is);
    auto cacheTransceiverConfig
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getCacheTransceiverConfig)>(is);
    auto gatherGenerationLogits
        = su::deserializeWithGetterType<decltype(&ExecutorConfig::getGatherGenerationLogits)>(is);
    auto promptTableOffloading = su::deserializeWithGetterType<decltype(&ExecutorConfig::getPromptTableOffloading)>(is);
    auto enableTrtOverlap = su::deserializeWithGetterType<decltype(&ExecutorConfig::getEnableTrtOverlap)>(is);

    return ExecutorConfig{maxBeamWidth, schedulerConfig, kvCacheConfig, enableChunkedContext, normalizeLogProbs,
        iterStatsMaxIterations, requestStatsMaxIterations, batchingType, maxBatchSize, maxNumTokens, parallelConfig,
        peftCacheConfig, std::nullopt, decodingConfig, useGpuDirectStorage, gpuWeightsPercent, maxQueueSize,
        extendedRuntimePerfKnobConfig, debugConfig, recvPollPeriodMs, maxSeqIdleMicroseconds, specDecConfig,
        guidedDecodingConfig, additionalModelOutputs, cacheTransceiverConfig, gatherGenerationLogits,
        promptTableOffloading, enableTrtOverlap};
}

size_t Serialization::serializedSize(ExecutorConfig const& executorConfig)
{
    TLLM_CHECK_WITH_INFO(!executorConfig.getLogitsPostProcessorConfig().has_value(),
        "Serialization of executorConfig with logitsPostProcessor is currently not supported.");

    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += su::serializedSize(executorConfig.getMaxBeamWidth());
    totalSize += su::serializedSize(executorConfig.getMaxBatchSize());
    totalSize += su::serializedSize(executorConfig.getMaxNumTokens());
    totalSize += su::serializedSize(executorConfig.getSchedulerConfig());
    totalSize += su::serializedSize(executorConfig.getKvCacheConfig());
    totalSize += su::serializedSize(executorConfig.getEnableChunkedContext());
    totalSize += su::serializedSize(executorConfig.getNormalizeLogProbs());
    totalSize += su::serializedSize(executorConfig.getIterStatsMaxIterations());
    totalSize += su::serializedSize(executorConfig.getRequestStatsMaxIterations());
    totalSize += su::serializedSize(executorConfig.getBatchingType());
    totalSize += su::serializedSize(executorConfig.getParallelConfig());
    totalSize += su::serializedSize(executorConfig.getPeftCacheConfig());
    totalSize += su::serializedSize(executorConfig.getDecodingConfig());
    totalSize += su::serializedSize(executorConfig.getUseGpuDirectStorage());
    totalSize += su::serializedSize(executorConfig.getGpuWeightsPercent());
    totalSize += su::serializedSize(executorConfig.getMaxQueueSize());
    totalSize += su::serializedSize(executorConfig.getExtendedRuntimePerfKnobConfig());
    totalSize += su::serializedSize(executorConfig.getDebugConfig());
    totalSize += su::serializedSize(executorConfig.getRecvPollPeriodMs());
    totalSize += su::serializedSize(executorConfig.getMaxSeqIdleMicroseconds());
    totalSize += su::serializedSize(executorConfig.getSpecDecConfig());
    totalSize += su::serializedSize(executorConfig.getGuidedDecodingConfig());
    totalSize += su::serializedSize(executorConfig.getAdditionalModelOutputs());
    totalSize += su::serializedSize(executorConfig.getCacheTransceiverConfig());
    totalSize += su::serializedSize(executorConfig.getGatherGenerationLogits());
    totalSize += su::serializedSize(executorConfig.getPromptTableOffloading());
    totalSize += su::serializedSize(executorConfig.getEnableTrtOverlap());

    return totalSize;
}

void Serialization::serialize(ExecutorConfig const& executorConfig, std::ostream& os)
{
    TLLM_CHECK_WITH_INFO(!executorConfig.getLogitsPostProcessorConfig().has_value(),
        "Serialization of executorConfig with logitsPostProcessor is currently not supported.");

    su::serialize(executorConfig.getMaxBeamWidth(), os);
    su::serialize(executorConfig.getMaxBatchSize(), os);
    su::serialize(executorConfig.getMaxNumTokens(), os);
    su::serialize(executorConfig.getSchedulerConfig(), os);
    su::serialize(executorConfig.getKvCacheConfig(), os);
    su::serialize(executorConfig.getEnableChunkedContext(), os);
    su::serialize(executorConfig.getNormalizeLogProbs(), os);
    su::serialize(executorConfig.getIterStatsMaxIterations(), os);
    su::serialize(executorConfig.getRequestStatsMaxIterations(), os);
    su::serialize(executorConfig.getBatchingType(), os);
    su::serialize(executorConfig.getParallelConfig(), os);
    su::serialize(executorConfig.getPeftCacheConfig(), os);
    su::serialize(executorConfig.getDecodingConfig(), os);
    su::serialize(executorConfig.getUseGpuDirectStorage(), os);
    su::serialize(executorConfig.getGpuWeightsPercent(), os);
    su::serialize(executorConfig.getMaxQueueSize(), os);
    su::serialize(executorConfig.getExtendedRuntimePerfKnobConfig(), os);
    su::serialize(executorConfig.getDebugConfig(), os);
    su::serialize(executorConfig.getRecvPollPeriodMs(), os);
    su::serialize(executorConfig.getMaxSeqIdleMicroseconds(), os);
    su::serialize(executorConfig.getSpecDecConfig(), os);
    su::serialize(executorConfig.getGuidedDecodingConfig(), os);
    su::serialize(executorConfig.getAdditionalModelOutputs(), os);
    su::serialize(executorConfig.getCacheTransceiverConfig(), os);
    su::serialize(executorConfig.getGatherGenerationLogits(), os);
    su::serialize(executorConfig.getPromptTableOffloading(), os);
    su::serialize(executorConfig.getEnableTrtOverlap(), os);
}

// KvCacheConfig
KvCacheConfig Serialization::deserializeKvCacheConfig(std::istream& is)
{
    auto enableBlockReuse = su::deserialize<bool>(is);
    auto enablePartialReuse = su::deserialize<bool>(is);
    auto copyOnPartialReuse = su::deserialize<bool>(is);
    auto maxTokens = su::deserialize<std::optional<SizeType32>>(is);
    auto maxAttentionWindowVec = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto sinkTokenLength = su::deserialize<std::optional<SizeType32>>(is);
    auto freeGpuMemoryFraction = su::deserialize<std::optional<FloatType>>(is);
    auto hostCacheSize = su::deserialize<std::optional<size_t>>(is);
    auto crossKvCacheFraction = su::deserialize<std::optional<FloatType>>(is);
    auto secondaryOffloadMinPriority = su::deserialize<std::optional<executor::RetentionPriority>>(is);
    auto eventBufferMaxSize = su::deserialize<size_t>(is);
    auto useUvm = su::deserialize<bool>(is);
    auto attentionDpEventsGatherPeriodMs = su::deserialize<SizeType32>(is);

    return KvCacheConfig{enableBlockReuse, maxTokens, maxAttentionWindowVec, sinkTokenLength, freeGpuMemoryFraction,
        hostCacheSize, crossKvCacheFraction, secondaryOffloadMinPriority, eventBufferMaxSize, enablePartialReuse,
        copyOnPartialReuse, useUvm, attentionDpEventsGatherPeriodMs};
}

void Serialization::serialize(KvCacheConfig const& kvCacheConfig, std::ostream& os)
{
    su::serialize(kvCacheConfig.getEnableBlockReuse(), os);
    su::serialize(kvCacheConfig.getEnablePartialReuse(), os);
    su::serialize(kvCacheConfig.getCopyOnPartialReuse(), os);
    su::serialize(kvCacheConfig.getMaxTokens(), os);
    su::serialize(kvCacheConfig.getMaxAttentionWindowVec(), os);
    su::serialize(kvCacheConfig.getSinkTokenLength(), os);
    su::serialize(kvCacheConfig.getFreeGpuMemoryFraction(), os);
    su::serialize(kvCacheConfig.getHostCacheSize(), os);
    su::serialize(kvCacheConfig.getCrossKvCacheFraction(), os);
    su::serialize(kvCacheConfig.getSecondaryOffloadMinPriority(), os);
    su::serialize(kvCacheConfig.getEventBufferMaxSize(), os);
    su::serialize(kvCacheConfig.getUseUvm(), os);
    su::serialize(kvCacheConfig.getAttentionDpEventsGatherPeriodMs(), os);
}

size_t Serialization::serializedSize(KvCacheConfig const& kvCacheConfig)
{

    size_t totalSize = 0;
    totalSize += su::serializedSize(kvCacheConfig.getEnableBlockReuse());
    totalSize += su::serializedSize(kvCacheConfig.getEnablePartialReuse());
    totalSize += su::serializedSize(kvCacheConfig.getCopyOnPartialReuse());
    totalSize += su::serializedSize(kvCacheConfig.getMaxTokens());
    totalSize += su::serializedSize(kvCacheConfig.getMaxAttentionWindowVec());
    totalSize += su::serializedSize(kvCacheConfig.getSinkTokenLength());
    totalSize += su::serializedSize(kvCacheConfig.getFreeGpuMemoryFraction());
    totalSize += su::serializedSize(kvCacheConfig.getHostCacheSize());
    totalSize += su::serializedSize(kvCacheConfig.getCrossKvCacheFraction());
    totalSize += su::serializedSize(kvCacheConfig.getSecondaryOffloadMinPriority());
    totalSize += su::serializedSize(kvCacheConfig.getEventBufferMaxSize());
    totalSize += su::serializedSize(kvCacheConfig.getUseUvm());
    totalSize += su::serializedSize(kvCacheConfig.getAttentionDpEventsGatherPeriodMs());
    return totalSize;
}

// DynamicBatchConfig
DynamicBatchConfig Serialization::deserializeDynamicBatchConfig(std::istream& is)
{
    auto enableBatchSizeTuning = su::deserialize<bool>(is);
    auto enableMaxNumTokensTuning = su::deserialize<bool>(is);
    auto dynamicBatchMovingAvgWindow = su::deserialize<SizeType32>(is);
    return DynamicBatchConfig{enableBatchSizeTuning, enableMaxNumTokensTuning, dynamicBatchMovingAvgWindow};
}

void Serialization::serialize(DynamicBatchConfig const& DynamicBatchConfig, std::ostream& os)
{
    su::serialize(DynamicBatchConfig.getEnableBatchSizeTuning(), os);
    su::serialize(DynamicBatchConfig.getEnableMaxNumTokensTuning(), os);
    su::serialize(DynamicBatchConfig.getDynamicBatchMovingAverageWindow(), os);
}

size_t Serialization::serializedSize(DynamicBatchConfig const& DynamicBatchConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(DynamicBatchConfig.getEnableBatchSizeTuning());
    totalSize += su::serializedSize(DynamicBatchConfig.getEnableMaxNumTokensTuning());
    totalSize += su::serializedSize(DynamicBatchConfig.getDynamicBatchMovingAverageWindow());
    return totalSize;
}

// SchedulerConfig
SchedulerConfig Serialization::deserializeSchedulerConfig(std::istream& is)
{
    auto capacitySchedulerPolicy = su::deserialize<CapacitySchedulerPolicy>(is);
    auto contextChunkingPolicy = su::deserialize<std::optional<ContextChunkingPolicy>>(is);
    auto dynamicBatchConfig = su::deserialize<std::optional<DynamicBatchConfig>>(is);
    return SchedulerConfig{capacitySchedulerPolicy, contextChunkingPolicy, dynamicBatchConfig};
}

void Serialization::serialize(SchedulerConfig const& schedulerConfig, std::ostream& os)
{
    su::serialize(schedulerConfig.getCapacitySchedulerPolicy(), os);
    su::serialize(schedulerConfig.getContextChunkingPolicy(), os);
    su::serialize(schedulerConfig.getDynamicBatchConfig(), os);
}

size_t Serialization::serializedSize(SchedulerConfig const& schedulerConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(schedulerConfig.getCapacitySchedulerPolicy());
    totalSize += su::serializedSize(schedulerConfig.getContextChunkingPolicy());
    totalSize += su::serializedSize(schedulerConfig.getDynamicBatchConfig());
    return totalSize;
}

// CacheTransceiverConfig
CacheTransceiverConfig Serialization::deserializeCacheTransceiverConfig(std::istream& is)
{
    auto backendType = su::deserialize<std::optional<CacheTransceiverConfig::BackendType>>(is);
    auto maxTokensInBuffer = su::deserialize<std::optional<size_t>>(is);
    auto kvTransferTimeoutMs = su::deserialize<std::optional<int>>(is);
    auto kvTransferSenderFutureTimeoutMs = su::deserialize<std::optional<int>>(is);
    return CacheTransceiverConfig{backendType, maxTokensInBuffer, kvTransferTimeoutMs, kvTransferSenderFutureTimeoutMs};
}

void Serialization::serialize(CacheTransceiverConfig const& cacheTransceiverConfig, std::ostream& os)
{
    su::serialize(cacheTransceiverConfig.getBackendType(), os);
    su::serialize(cacheTransceiverConfig.getMaxTokensInBuffer(), os);
    su::serialize(cacheTransceiverConfig.getKvTransferTimeoutMs(), os);
    su::serialize(cacheTransceiverConfig.getKvTransferSenderFutureTimeoutMs(), os);
}

size_t Serialization::serializedSize(CacheTransceiverConfig const& cacheTransceiverConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(cacheTransceiverConfig.getBackendType());
    totalSize += su::serializedSize(cacheTransceiverConfig.getMaxTokensInBuffer());
    totalSize += su::serializedSize(cacheTransceiverConfig.getKvTransferTimeoutMs());
    totalSize += su::serializedSize(cacheTransceiverConfig.getKvTransferSenderFutureTimeoutMs());
    return totalSize;
}

// ExtendedRuntimePerfKnobConfig
ExtendedRuntimePerfKnobConfig Serialization::deserializeExtendedRuntimePerfKnobConfig(std::istream& is)
{
    auto multiBlockMode = su::deserialize<bool>(is);
    auto enableContextFMHAFP32Acc = su::deserialize<bool>(is);
    auto cudaGraphMode = su::deserialize<bool>(is);
    auto cudaGraphCacheSize = su::deserialize<SizeType32>(is);
    return ExtendedRuntimePerfKnobConfig{multiBlockMode, enableContextFMHAFP32Acc, cudaGraphMode, cudaGraphCacheSize};
}

void Serialization::serialize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, std::ostream& os)
{
    su::serialize(extendedRuntimePerfKnobConfig.getMultiBlockMode(), os);
    su::serialize(extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc(), os);
    su::serialize(extendedRuntimePerfKnobConfig.getCudaGraphMode(), os);
    su::serialize(extendedRuntimePerfKnobConfig.getCudaGraphCacheSize(), os);
}

size_t Serialization::serializedSize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getMultiBlockMode());
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc());
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getCudaGraphMode());
    totalSize += su::serializedSize(extendedRuntimePerfKnobConfig.getCudaGraphCacheSize());
    return totalSize;
}

// ParallelConfig
ParallelConfig Serialization::deserializeParallelConfig(std::istream& is)
{
    auto commType = su::deserialize<CommunicationType>(is);
    auto commMode = su::deserialize<CommunicationMode>(is);
    auto deviceIds = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto participantids = su::deserialize<std::optional<std::vector<SizeType32>>>(is);
    auto orchestratorConfig = su::deserialize<std::optional<OrchestratorConfig>>(is);
    auto numNodes = su::deserialize<std::optional<SizeType32>>(is);

    return ParallelConfig{commType, commMode, deviceIds, participantids, orchestratorConfig, numNodes};
}

void Serialization::serialize(ParallelConfig const& parallelConfig, std::ostream& os)
{
    su::serialize(parallelConfig.getCommunicationType(), os);
    su::serialize(parallelConfig.getCommunicationMode(), os);
    su::serialize(parallelConfig.getDeviceIds(), os);
    su::serialize(parallelConfig.getParticipantIds(), os);
    su::serialize(parallelConfig.getOrchestratorConfig(), os);
    su::serialize(parallelConfig.getNumNodes(), os);
}

size_t Serialization::serializedSize(ParallelConfig const& parallelConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(parallelConfig.getCommunicationType());
    totalSize += su::serializedSize(parallelConfig.getCommunicationMode());
    totalSize += su::serializedSize(parallelConfig.getDeviceIds());
    totalSize += su::serializedSize(parallelConfig.getParticipantIds());
    totalSize += su::serializedSize(parallelConfig.getOrchestratorConfig());
    totalSize += su::serializedSize(parallelConfig.getNumNodes());
    return totalSize;
}

// PeftCacheConfig
PeftCacheConfig Serialization::deserializePeftCacheConfig(std::istream& is)
{
    auto numHostModuleLayer = su::deserialize<SizeType32>(is);
    auto numDeviceModuleLayer = su::deserialize<SizeType32>(is);
    auto optimalAdapterSize = su::deserialize<SizeType32>(is);
    auto maxAdapterSize = su::deserialize<SizeType32>(is);
    auto numPutWorkers = su::deserialize<SizeType32>(is);
    auto numEnsureWorkers = su::deserialize<SizeType32>(is);
    auto numCopyStreams = su::deserialize<SizeType32>(is);
    auto maxPagesPerBlockHost = su::deserialize<SizeType32>(is);
    auto maxPagesPerBlockDevice = su::deserialize<SizeType32>(is);
    auto deviceCachePercent = su::deserialize<std::optional<FloatType>>(is);
    auto hostCacheSize = su::deserialize<std::optional<size_t>>(is);

    return PeftCacheConfig{numHostModuleLayer, numDeviceModuleLayer, optimalAdapterSize, maxAdapterSize, numPutWorkers,
        numEnsureWorkers, numCopyStreams, maxPagesPerBlockHost, maxPagesPerBlockDevice, deviceCachePercent,
        hostCacheSize};
}

void Serialization::serialize(PeftCacheConfig const& peftCacheConfig, std::ostream& os)
{
    su::serialize(peftCacheConfig.getNumHostModuleLayer(), os);
    su::serialize(peftCacheConfig.getNumDeviceModuleLayer(), os);
    su::serialize(peftCacheConfig.getOptimalAdapterSize(), os);
    su::serialize(peftCacheConfig.getMaxAdapterSize(), os);
    su::serialize(peftCacheConfig.getNumPutWorkers(), os);
    su::serialize(peftCacheConfig.getNumEnsureWorkers(), os);
    su::serialize(peftCacheConfig.getNumCopyStreams(), os);
    su::serialize(peftCacheConfig.getMaxPagesPerBlockHost(), os);
    su::serialize(peftCacheConfig.getMaxPagesPerBlockDevice(), os);
    su::serialize(peftCacheConfig.getDeviceCachePercent(), os);
    su::serialize(peftCacheConfig.getHostCacheSize(), os);
}

size_t Serialization::serializedSize(PeftCacheConfig const& peftCacheConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(peftCacheConfig.getNumHostModuleLayer());
    totalSize += su::serializedSize(peftCacheConfig.getNumDeviceModuleLayer());
    totalSize += su::serializedSize(peftCacheConfig.getOptimalAdapterSize());
    totalSize += su::serializedSize(peftCacheConfig.getMaxAdapterSize());
    totalSize += su::serializedSize(peftCacheConfig.getNumPutWorkers());
    totalSize += su::serializedSize(peftCacheConfig.getNumEnsureWorkers());
    totalSize += su::serializedSize(peftCacheConfig.getNumCopyStreams());
    totalSize += su::serializedSize(peftCacheConfig.getMaxPagesPerBlockHost());
    totalSize += su::serializedSize(peftCacheConfig.getMaxPagesPerBlockDevice());
    totalSize += su::serializedSize(peftCacheConfig.getDeviceCachePercent());
    totalSize += su::serializedSize(peftCacheConfig.getHostCacheSize());
    return totalSize;
}

// OrchestratorConfig
OrchestratorConfig Serialization::deserializeOrchestratorConfig(std::istream& is)
{
    auto isOrchestrator = su::deserialize<bool>(is);
    auto path = su::deserialize<std::string>(is);
    auto spawnProcesses = su::deserialize<bool>(is);
    // Note we ignore mpiComm since we don't need to exchange it
    return OrchestratorConfig{isOrchestrator, path, nullptr, spawnProcesses};
}

void Serialization::serialize(OrchestratorConfig const& orchestratorConfig, std::ostream& os)
{
    su::serialize(orchestratorConfig.getIsOrchestrator(), os);
    su::serialize(orchestratorConfig.getWorkerExecutablePath(), os);
    su::serialize(orchestratorConfig.getSpawnProcesses(), os);
}

size_t Serialization::serializedSize(OrchestratorConfig const& orchestratorConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(orchestratorConfig.getIsOrchestrator());
    totalSize += su::serializedSize(orchestratorConfig.getWorkerExecutablePath());
    totalSize += su::serializedSize(orchestratorConfig.getSpawnProcesses());
    return totalSize;
}

// DecodingMode
DecodingMode Serialization::deserializeDecodingMode(std::istream& is)
{
    auto mode = su::deserialize<DecodingMode::UnderlyingType>(is);

    return DecodingMode{mode};
}

void Serialization::serialize(DecodingMode const& decodingMode, std::ostream& os)
{
    su::serialize(decodingMode.getState(), os);
}

size_t Serialization::serializedSize(DecodingMode const& decodingMode)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(decodingMode.getState());
    return totalSize;
}

// LookaheadDecodingConfig
LookaheadDecodingConfig Serialization::deserializeLookaheadDecodingConfig(std::istream& is)
{
    auto ngramSize = su::deserialize<SizeType32>(is);
    auto windowSize = su::deserialize<SizeType32>(is);
    auto verificationSetSize = su::deserialize<SizeType32>(is);

    return LookaheadDecodingConfig{windowSize, ngramSize, verificationSetSize};
}

void Serialization::serialize(LookaheadDecodingConfig const& lookaheadDecodingConfig, std::ostream& os)
{
    su::serialize(lookaheadDecodingConfig.getNgramSize(), os);
    su::serialize(lookaheadDecodingConfig.getWindowSize(), os);
    su::serialize(lookaheadDecodingConfig.getVerificationSetSize(), os);
}

size_t Serialization::serializedSize(LookaheadDecodingConfig const& lookaheadDecodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(lookaheadDecodingConfig.getNgramSize());
    totalSize += su::serializedSize(lookaheadDecodingConfig.getWindowSize());
    totalSize += su::serializedSize(lookaheadDecodingConfig.getVerificationSetSize());
    return totalSize;
}

// EagleConfig
EagleConfig Serialization::deserializeEagleConfig(std::istream& is)
{
    auto eagleChoices = su::deserialize<std::optional<EagleChoices>>(is);
    auto isGreedySampling = su::deserialize<bool>(is);
    auto posteriorThreshold = su::deserialize<std::optional<float>>(is);
    auto useDynamicTree = su::deserialize<bool>(is);
    auto dynamicTreeMaxTopK = su::deserialize<std::optional<SizeType32>>(is);

    return EagleConfig{eagleChoices, isGreedySampling, posteriorThreshold, useDynamicTree, dynamicTreeMaxTopK};
}

void Serialization::serialize(EagleConfig const& eagleConfig, std::ostream& os)
{
    su::serialize(eagleConfig.getEagleChoices(), os);
    su::serialize(eagleConfig.isGreedySampling(), os);
    su::serialize(eagleConfig.getPosteriorThreshold(), os);
    su::serialize(eagleConfig.useDynamicTree(), os);
    su::serialize(eagleConfig.getDynamicTreeMaxTopK(), os);
}

size_t Serialization::serializedSize(EagleConfig const& eagleConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(eagleConfig.getEagleChoices());
    totalSize += su::serializedSize(eagleConfig.isGreedySampling());
    totalSize += su::serializedSize(eagleConfig.getPosteriorThreshold());
    totalSize += su::serializedSize(eagleConfig.useDynamicTree());
    totalSize += su::serializedSize(eagleConfig.getDynamicTreeMaxTopK());

    return totalSize;
}

// SpeculativeDecodingConfig
SpeculativeDecodingConfig Serialization::deserializeSpeculativeDecodingConfig(std::istream& is)
{
    auto fastLogits = su::deserialize<decltype(SpeculativeDecodingConfig::fastLogits)>(is);
    return SpeculativeDecodingConfig(fastLogits);
}

void Serialization::serialize(SpeculativeDecodingConfig const& specDecConfig, std::ostream& os)
{
    su::serialize(specDecConfig.fastLogits, os);
}

size_t Serialization::serializedSize(SpeculativeDecodingConfig const& specDecConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(specDecConfig.fastLogits);
    return totalSize;
}

// GuidedDecodingConfig
GuidedDecodingConfig Serialization::deserializeGuidedDecodingConfig(std::istream& is)
{
    auto backend = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getBackend)>(is);
    auto encodedVocab = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getEncodedVocab)>(is);
    auto tokenizerStr = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getTokenizerStr)>(is);
    auto stopTokenIds = su::deserializeWithGetterType<decltype(&GuidedDecodingConfig::getStopTokenIds)>(is);
    return GuidedDecodingConfig(backend, encodedVocab, tokenizerStr, stopTokenIds);
}

void Serialization::serialize(GuidedDecodingConfig const& guidedDecodingConfig, std::ostream& os)
{
    su::serialize(guidedDecodingConfig.getBackend(), os);
    su::serialize(guidedDecodingConfig.getEncodedVocab(), os);
    su::serialize(guidedDecodingConfig.getTokenizerStr(), os);
    su::serialize(guidedDecodingConfig.getStopTokenIds(), os);
}

size_t Serialization::serializedSize(GuidedDecodingConfig const& guidedDecodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(guidedDecodingConfig.getBackend());
    totalSize += su::serializedSize(guidedDecodingConfig.getEncodedVocab());
    totalSize += su::serializedSize(guidedDecodingConfig.getTokenizerStr());
    totalSize += su::serializedSize(guidedDecodingConfig.getStopTokenIds());
    return totalSize;
}

// GuidedDecodingParams
GuidedDecodingParams Serialization::deserializeGuidedDecodingParams(std::istream& is)
{
    auto guideType = su::deserializeWithGetterType<decltype(&GuidedDecodingParams::getGuideType)>(is);
    auto guide = su::deserializeWithGetterType<decltype(&GuidedDecodingParams::getGuide)>(is);
    return GuidedDecodingParams(guideType, guide);
}

void Serialization::serialize(GuidedDecodingParams const& guidedDecodingParams, std::ostream& os)
{
    su::serialize(guidedDecodingParams.getGuideType(), os);
    su::serialize(guidedDecodingParams.getGuide(), os);
}

size_t Serialization::serializedSize(GuidedDecodingParams const& guidedDecodingParams)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(guidedDecodingParams.getGuideType());
    totalSize += su::serializedSize(guidedDecodingParams.getGuide());
    return totalSize;
}

// KV Cache Retention Config
std::optional<size_t> durationToInt(std::optional<std::chrono::milliseconds> const& duration)
{
    return duration.has_value() ? std::optional(static_cast<size_t>(duration->count())) : std::nullopt;
}

std::optional<std::chrono::milliseconds> intToDuration(std::optional<size_t> const& duration)
{
    return duration.has_value() ? std::optional(std::chrono::milliseconds(*duration)) : std::nullopt;
}

KvCacheRetentionConfig Serialization::deserializeKvCacheRetentionConfig(std::istream& is)
{
    auto tokenRangeRetentionPriorities
        = su::deserialize<std::vector<executor::KvCacheRetentionConfig::TokenRangeRetentionConfig>>(is);
    auto decodePriority = su::deserialize<executor::RetentionPriority>(is);
    auto decodeDurationMs = intToDuration(su::deserialize<std::optional<size_t>>(is));
    auto transferMode = su::deserialize<executor::KvCacheTransferMode>(is);
    auto directory = su::deserialize<std::string>(is);

    return KvCacheRetentionConfig{
        tokenRangeRetentionPriorities, decodePriority, decodeDurationMs, transferMode, directory};
}

void Serialization::serialize(KvCacheRetentionConfig const& kvCacheRetentionConfig, std::ostream& os)
{
    su::serialize(kvCacheRetentionConfig.getTokenRangeRetentionConfigs(), os);
    su::serialize(kvCacheRetentionConfig.getDecodeRetentionPriority(), os);
    su::serialize(durationToInt(kvCacheRetentionConfig.getDecodeDurationMs()), os);
    su::serialize(kvCacheRetentionConfig.getTransferMode(), os);
    su::serialize(kvCacheRetentionConfig.getDirectory(), os);
}

size_t Serialization::serializedSize(KvCacheRetentionConfig const& config)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(config.getTokenRangeRetentionConfigs());
    totalSize += su::serializedSize(config.getDecodeRetentionPriority());
    totalSize += su::serializedSize(durationToInt(config.getDecodeDurationMs()));
    totalSize += su::serializedSize(config.getTransferMode());
    totalSize += su::serializedSize(config.getDirectory());
    return totalSize;
}

// TokenRangeRetentionConfig
KvCacheRetentionConfig::TokenRangeRetentionConfig Serialization::deserializeTokenRangeRetentionConfig(std::istream& is)
{
    auto tokenStart = su::deserialize<SizeType32>(is);
    auto tokenEnd = su::deserialize<std::optional<SizeType32>>(is);
    auto priority = static_cast<RetentionPriority>(su::deserialize<RetentionPriority>(is));
    auto durationMs = intToDuration(su::deserialize<std::optional<size_t>>(is));

    return KvCacheRetentionConfig::TokenRangeRetentionConfig(tokenStart, tokenEnd, priority, durationMs);
}

void Serialization::serialize(
    KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig, std::ostream& os)
{
    su::serialize(tokenRangeRetentionConfig.tokenStart, os);
    su::serialize(tokenRangeRetentionConfig.tokenEnd, os);
    su::serialize(tokenRangeRetentionConfig.priority, os);
    su::serialize(durationToInt(tokenRangeRetentionConfig.durationMs), os);
}

size_t Serialization::serializedSize(KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(tokenRangeRetentionConfig.tokenStart);
    totalSize += su::serializedSize(tokenRangeRetentionConfig.tokenEnd);
    totalSize += su::serializedSize(static_cast<RetentionPriority>(tokenRangeRetentionConfig.priority));
    totalSize += su::serializedSize(durationToInt(tokenRangeRetentionConfig.durationMs));
    return totalSize;
}

// DecodingConfig
DecodingConfig Serialization::deserializeDecodingConfig(std::istream& is)
{
    auto decodingMode = su::deserialize<std::optional<DecodingMode>>(is);
    auto lookaheadDecodingConfig = su::deserialize<std::optional<LookaheadDecodingConfig>>(is);
    auto medusaChoices = su::deserialize<std::optional<MedusaChoices>>(is);
    auto eagleConfig = su::deserialize<std::optional<EagleConfig>>(is);

    return DecodingConfig{decodingMode, lookaheadDecodingConfig, medusaChoices, eagleConfig};
}

void Serialization::serialize(DecodingConfig const& decodingConfig, std::ostream& os)
{
    su::serialize(decodingConfig.getDecodingMode(), os);
    su::serialize(decodingConfig.getLookaheadDecodingConfig(), os);
    su::serialize(decodingConfig.getMedusaChoices(), os);
    su::serialize(decodingConfig.getEagleConfig(), os);
}

size_t Serialization::serializedSize(DecodingConfig const& decodingConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(decodingConfig.getDecodingMode());
    totalSize += su::serializedSize(decodingConfig.getLookaheadDecodingConfig());
    totalSize += su::serializedSize(decodingConfig.getMedusaChoices());
    totalSize += su::serializedSize(decodingConfig.getEagleConfig());
    return totalSize;
}

// DebugConfig
DebugConfig Serialization::deserializeDebugConfig(std::istream& is)
{
    auto debugInputTensors = su::deserializeWithGetterType<decltype(&DebugConfig::getDebugInputTensors)>(is);
    auto debugOutputTensors = su::deserializeWithGetterType<decltype(&DebugConfig::getDebugOutputTensors)>(is);
    auto debugTensorNames = su::deserialize<std::remove_cv_t<
        std::remove_reference_t<std::invoke_result_t<decltype(&DebugConfig::getDebugTensorNames), DebugConfig>>>>(is);
    auto debugTensorsMaxIterations
        = su::deserializeWithGetterType<decltype(&DebugConfig::getDebugTensorsMaxIterations)>(is);

    return DebugConfig{debugInputTensors, debugOutputTensors, debugTensorNames, debugTensorsMaxIterations};
}

void Serialization::serialize(DebugConfig const& debugConfig, std::ostream& os)
{
    su::serialize(debugConfig.getDebugInputTensors(), os);
    su::serialize(debugConfig.getDebugOutputTensors(), os);
    su::serialize(debugConfig.getDebugTensorNames(), os);
    su::serialize(debugConfig.getDebugTensorsMaxIterations(), os);
}

size_t Serialization::serializedSize(DebugConfig const& debugConfig)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(debugConfig.getDebugInputTensors());
    totalSize += su::serializedSize(debugConfig.getDebugOutputTensors());
    totalSize += su::serializedSize(debugConfig.getDebugTensorNames());
    totalSize += su::serializedSize(debugConfig.getDebugTensorsMaxIterations());
    return totalSize;
}

// KvCacheStats
KvCacheStats Serialization::deserializeKvCacheStats(std::istream& is)
{
    auto maxNumBlocks = su::deserialize<SizeType32>(is);
    auto freeNumBlocks = su::deserialize<SizeType32>(is);
    auto usedNumBlocks = su::deserialize<SizeType32>(is);
    auto tokensPerBlock = su::deserialize<SizeType32>(is);
    auto allocTotalBlocks = su::deserialize<SizeType32>(is);
    auto allocNewBlocks = su::deserialize<SizeType32>(is);
    auto reusedBlocks = su::deserialize<SizeType32>(is);
    auto missedBlocks = su::deserialize<SizeType32>(is);
    auto cacheHitRate = su::deserialize<float>(is);

    return KvCacheStats{maxNumBlocks, freeNumBlocks, usedNumBlocks, tokensPerBlock, allocTotalBlocks, allocNewBlocks,
        reusedBlocks, missedBlocks, cacheHitRate};
}

void Serialization::serialize(KvCacheStats const& kvCacheStats, std::ostream& os)
{
    su::serialize(kvCacheStats.maxNumBlocks, os);
    su::serialize(kvCacheStats.freeNumBlocks, os);
    su::serialize(kvCacheStats.usedNumBlocks, os);
    su::serialize(kvCacheStats.tokensPerBlock, os);
    su::serialize(kvCacheStats.allocTotalBlocks, os);
    su::serialize(kvCacheStats.allocNewBlocks, os);
    su::serialize(kvCacheStats.reusedBlocks, os);
    su::serialize(kvCacheStats.missedBlocks, os);
    su::serialize(kvCacheStats.cacheHitRate, os);
}

size_t Serialization::serializedSize(KvCacheStats const& kvCacheStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(kvCacheStats.maxNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.freeNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.usedNumBlocks);
    totalSize += su::serializedSize(kvCacheStats.tokensPerBlock);
    totalSize += su::serializedSize(kvCacheStats.allocTotalBlocks);
    totalSize += su::serializedSize(kvCacheStats.allocNewBlocks);
    totalSize += su::serializedSize(kvCacheStats.reusedBlocks);
    totalSize += su::serializedSize(kvCacheStats.missedBlocks);
    totalSize += su::serializedSize(kvCacheStats.cacheHitRate);
    return totalSize;
}

// StaticBatchingStats
StaticBatchingStats Serialization::deserializeStaticBatchingStats(std::istream& is)
{
    auto numScheduledRequests = su::deserialize<SizeType32>(is);
    auto numContextRequests = su::deserialize<SizeType32>(is);
    auto numCtxTokens = su::deserialize<SizeType32>(is);
    auto numGenTokens = su::deserialize<SizeType32>(is);
    auto emptyGenSlots = su::deserialize<SizeType32>(is);
    return StaticBatchingStats{numScheduledRequests, numContextRequests, numCtxTokens, numGenTokens, emptyGenSlots};
}

void Serialization::serialize(StaticBatchingStats const& staticBatchingStats, std::ostream& os)
{
    su::serialize(staticBatchingStats.numScheduledRequests, os);
    su::serialize(staticBatchingStats.numContextRequests, os);
    su::serialize(staticBatchingStats.numCtxTokens, os);
    su::serialize(staticBatchingStats.numGenTokens, os);
    su::serialize(staticBatchingStats.emptyGenSlots, os);
}

size_t Serialization::serializedSize(StaticBatchingStats const& staticBatchingStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(staticBatchingStats.numScheduledRequests);
    totalSize += su::serializedSize(staticBatchingStats.numContextRequests);
    totalSize += su::serializedSize(staticBatchingStats.numCtxTokens);
    totalSize += su::serializedSize(staticBatchingStats.numGenTokens);
    totalSize += su::serializedSize(staticBatchingStats.emptyGenSlots);
    return totalSize;
}

// InflightBatchingStats
InflightBatchingStats Serialization::deserializeInflightBatchingStats(std::istream& is)
{
    auto numScheduledRequests = su::deserialize<SizeType32>(is);
    auto numContextRequests = su::deserialize<SizeType32>(is);
    auto numGenRequests = su::deserialize<SizeType32>(is);
    auto numPausedRequests = su::deserialize<SizeType32>(is);
    auto numCtxTokens = su::deserialize<SizeType32>(is);
    auto microBatchId = su::deserialize<SizeType32>(is);
    auto avgNumDecodedTokensPerIter = su::deserialize<float>(is);
    auto numCtxKvTokens = su::deserialize<SizeType32>(is);
    auto numGenKvTokens = su::deserialize<SizeType32>(is);
    auto numQueuedContextRequests = su::deserialize<SizeType32>(is);
    auto numQueuedCtxTokens = su::deserialize<SizeType32>(is);
    auto numQueuedGenRequests = su::deserialize<SizeType32>(is);
    auto numQueuedGenKvTokens = su::deserialize<SizeType32>(is);
    auto numPausedKvTokens = su::deserialize<SizeType32>(is);
    return InflightBatchingStats{numScheduledRequests, numContextRequests, numGenRequests, numPausedRequests,
        numCtxTokens, microBatchId, avgNumDecodedTokensPerIter, numCtxKvTokens, numGenKvTokens,
        numQueuedContextRequests, numQueuedCtxTokens, numQueuedGenRequests, numQueuedGenKvTokens, numPausedKvTokens};
}

void Serialization::serialize(InflightBatchingStats const& inflightBatchingStats, std::ostream& os)
{
    su::serialize(inflightBatchingStats.numScheduledRequests, os);
    su::serialize(inflightBatchingStats.numContextRequests, os);
    su::serialize(inflightBatchingStats.numGenRequests, os);
    su::serialize(inflightBatchingStats.numPausedRequests, os);
    su::serialize(inflightBatchingStats.numCtxTokens, os);
    su::serialize(inflightBatchingStats.microBatchId, os);
    su::serialize(inflightBatchingStats.avgNumDecodedTokensPerIter, os);
    su::serialize(inflightBatchingStats.numCtxKvTokens, os);
    su::serialize(inflightBatchingStats.numGenKvTokens, os);
    su::serialize(inflightBatchingStats.numQueuedContextRequests, os);
    su::serialize(inflightBatchingStats.numQueuedCtxTokens, os);
    su::serialize(inflightBatchingStats.numQueuedGenRequests, os);
    su::serialize(inflightBatchingStats.numQueuedGenKvTokens, os);
    su::serialize(inflightBatchingStats.numPausedKvTokens, os);
}

size_t Serialization::serializedSize(InflightBatchingStats const& inflightBatchingStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(inflightBatchingStats.numScheduledRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numContextRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numGenRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numPausedRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numCtxTokens);
    totalSize += su::serializedSize(inflightBatchingStats.microBatchId);
    totalSize += su::serializedSize(inflightBatchingStats.avgNumDecodedTokensPerIter);
    totalSize += su::serializedSize(inflightBatchingStats.numCtxKvTokens);
    totalSize += su::serializedSize(inflightBatchingStats.numGenKvTokens);
    totalSize += su::serializedSize(inflightBatchingStats.numQueuedContextRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numQueuedCtxTokens);
    totalSize += su::serializedSize(inflightBatchingStats.numQueuedGenRequests);
    totalSize += su::serializedSize(inflightBatchingStats.numQueuedGenKvTokens);
    totalSize += su::serializedSize(inflightBatchingStats.numPausedKvTokens);
    return totalSize;
}

// SpecDecodingStats
SpecDecodingStats Serialization::deserializeSpecDecodingStats(std::istream& is)
{
    auto numDraftTokens = su::deserialize<SizeType64>(is);
    auto numAcceptedTokens = su::deserialize<SizeType64>(is);
    auto numRequestsWithDraftTokens = su::deserialize<SizeType64>(is);
    auto acceptanceLength = su::deserialize<double>(is);
    auto iterLatencyMS = su::deserialize<double>(is);
    auto draftOverhead = su::deserialize<double>(is);

    return SpecDecodingStats{
        numDraftTokens, numAcceptedTokens, numRequestsWithDraftTokens, acceptanceLength, iterLatencyMS, draftOverhead};
}

void Serialization::serialize(SpecDecodingStats const& state, std::ostream& os)
{
    su::serialize(state.numDraftTokens, os);
    su::serialize(state.numAcceptedTokens, os);
    su::serialize(state.numRequestsWithDraftTokens, os);
    su::serialize(state.acceptanceLength, os);
    su::serialize(state.iterLatencyMS, os);
    su::serialize(state.draftOverhead, os);
}

size_t Serialization::serializedSize(SpecDecodingStats const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.numDraftTokens);
    totalSize += su::serializedSize(state.numAcceptedTokens);
    totalSize += su::serializedSize(state.numRequestsWithDraftTokens);
    totalSize += su::serializedSize(state.acceptanceLength);
    totalSize += su::serializedSize(state.iterLatencyMS);
    totalSize += su::serializedSize(state.draftOverhead);
    return totalSize;
}

// IterationStats

IterationStats Serialization::deserializeIterationStats(std::istream& is)
{
    IterationStats iterStats;
    auto timestamp = su::deserialize<std::string>(is);
    auto iter = su::deserialize<IterationType>(is);
    auto iterLatencyMS = su::deserialize<double>(is);
    auto newActiveRequestsQueueLatencyMS = su::deserialize<double>(is);
    auto numNewActiveRequests = su::deserialize<SizeType32>(is);
    auto numActiveRequests = su::deserialize<SizeType32>(is);
    auto numQueuedRequests = su::deserialize<SizeType32>(is);
    auto numCompletedRequests = su::deserialize<SizeType32>(is);
    auto maxNumActiveRequests = su::deserialize<SizeType32>(is);
    auto maxBatchSizeStatic = su::deserialize<SizeType32>(is);
    auto maxBatchSizeTunerRecommended = su::deserialize<SizeType32>(is);
    auto maxBatchSizeRuntime = su::deserialize<SizeType32>(is);
    auto maxNumTokensStatic = su::deserialize<SizeType32>(is);
    auto maxNumTokensTunerRecommended = su::deserialize<SizeType32>(is);
    auto maxNumTokensRuntime = su::deserialize<SizeType32>(is);
    auto gpuMemUsage = su::deserialize<size_t>(is);
    auto cpuMemUsage = su::deserialize<size_t>(is);
    auto pinnedMemUsage = su::deserialize<size_t>(is);
    auto kvCacheStats = su::deserialize<std::optional<KvCacheStats>>(is);
    auto crossKvCacheStats = su::deserialize<std::optional<KvCacheStats>>(is);
    auto staticBatchingStats = su::deserialize<std::optional<StaticBatchingStats>>(is);
    auto inflightBatchingStats = su::deserialize<std::optional<InflightBatchingStats>>(is);
    auto specDecodingStats = su::deserialize<std::optional<SpecDecodingStats>>(is);

    return IterationStats{timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS, numNewActiveRequests,
        numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests, maxBatchSizeStatic,
        maxBatchSizeTunerRecommended, maxBatchSizeRuntime, maxNumTokensStatic, maxNumTokensTunerRecommended,
        maxNumTokensRuntime, gpuMemUsage, cpuMemUsage, pinnedMemUsage, kvCacheStats, crossKvCacheStats,
        staticBatchingStats, inflightBatchingStats, specDecodingStats};
}

IterationStats Serialization::deserializeIterationStats(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);

    return Serialization::deserializeIterationStats(is);
}

size_t Serialization::serializedSize(IterationStats const& iterStats)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;

    totalSize += su::serializedSize(iterStats.timestamp);
    totalSize += su::serializedSize(iterStats.iter);
    totalSize += su::serializedSize(iterStats.iterLatencyMS);
    totalSize += su::serializedSize(iterStats.newActiveRequestsQueueLatencyMS);
    totalSize += su::serializedSize(iterStats.numNewActiveRequests);
    totalSize += su::serializedSize(iterStats.numActiveRequests);
    totalSize += su::serializedSize(iterStats.numQueuedRequests);
    totalSize += su::serializedSize(iterStats.numCompletedRequests);
    totalSize += su::serializedSize(iterStats.maxNumActiveRequests);
    totalSize += su::serializedSize(iterStats.maxBatchSizeStatic);
    totalSize += su::serializedSize(iterStats.maxBatchSizeTunerRecommended);
    totalSize += su::serializedSize(iterStats.maxBatchSizeRuntime);
    totalSize += su::serializedSize(iterStats.maxNumTokensStatic);
    totalSize += su::serializedSize(iterStats.maxNumTokensTunerRecommended);
    totalSize += su::serializedSize(iterStats.maxNumTokensRuntime);
    totalSize += su::serializedSize(iterStats.gpuMemUsage);
    totalSize += su::serializedSize(iterStats.cpuMemUsage);
    totalSize += su::serializedSize(iterStats.pinnedMemUsage);
    totalSize += su::serializedSize(iterStats.kvCacheStats);
    totalSize += su::serializedSize(iterStats.crossKvCacheStats);
    totalSize += su::serializedSize(iterStats.staticBatchingStats);
    totalSize += su::serializedSize(iterStats.inflightBatchingStats);
    totalSize += su::serializedSize(iterStats.specDecodingStats);

    return totalSize;
}

void Serialization::serialize(IterationStats const& iterStats, std::ostream& os)
{
    su::serialize(iterStats.timestamp, os);
    su::serialize(iterStats.iter, os);
    su::serialize(iterStats.iterLatencyMS, os);
    su::serialize(iterStats.newActiveRequestsQueueLatencyMS, os);
    su::serialize(iterStats.numNewActiveRequests, os);
    su::serialize(iterStats.numActiveRequests, os);
    su::serialize(iterStats.numQueuedRequests, os);
    su::serialize(iterStats.numCompletedRequests, os);
    su::serialize(iterStats.maxNumActiveRequests, os);
    su::serialize(iterStats.maxBatchSizeStatic, os);
    su::serialize(iterStats.maxBatchSizeTunerRecommended, os);
    su::serialize(iterStats.maxBatchSizeRuntime, os);
    su::serialize(iterStats.maxNumTokensStatic, os);
    su::serialize(iterStats.maxNumTokensTunerRecommended, os);
    su::serialize(iterStats.maxNumTokensRuntime, os);
    su::serialize(iterStats.gpuMemUsage, os);
    su::serialize(iterStats.cpuMemUsage, os);
    su::serialize(iterStats.pinnedMemUsage, os);
    su::serialize(iterStats.kvCacheStats, os);
    su::serialize(iterStats.crossKvCacheStats, os);
    su::serialize(iterStats.staticBatchingStats, os);
    su::serialize(iterStats.inflightBatchingStats, os);
    su::serialize(iterStats.specDecodingStats, os);
}

std::vector<char> Serialization::serialize(IterationStats const& iterStats)
{
    auto totalSize = Serialization::serializedSize(iterStats);
    std::vector<char> buffer(totalSize);

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    Serialization::serialize(iterStats, os);

    return buffer;
}

std::vector<IterationStats> Serialization::deserializeIterationStatsVec(std::vector<char>& buffer)
{
    std::vector<IterationStats> iterStatsVec;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numIterStats = su::deserialize<std::size_t>(is);
    for (std::size_t iterStats = 0; iterStats < numIterStats; ++iterStats)
    {
        iterStatsVec.emplace_back(Serialization::deserializeIterationStats(is));
    }
    return iterStatsVec;
}

std::vector<char> Serialization::serialize(std::vector<IterationStats> const& iterStatsVec)
{
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& iterStats : iterStatsVec)
    {
        totalSize += su::serializedSize(iterStats);
    }
    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);
    su::serialize(iterStatsVec.size(), os);
    for (auto const& iterStats : iterStatsVec)
    {
        su::serialize(iterStats, os);
    }
    return buffer;
}

// DistServinigState

DisServingRequestStats Serialization::deserializeDisServingRequestStats(std::istream& is)
{
    auto kvCacheTransferMs = su::deserialize<double>(is);
    auto kvCacheSize = su::deserialize<size_t>(is);
    return DisServingRequestStats{kvCacheTransferMs, kvCacheSize};
}

void Serialization::serialize(DisServingRequestStats const& stats, std::ostream& os)
{
    su::serialize(stats.kvCacheTransferMS, os);
    su::serialize(stats.kvCacheSize, os);
}

size_t Serialization::serializedSize(DisServingRequestStats const& disServingRequestStats)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(disServingRequestStats.kvCacheTransferMS);
    totalSize += su::serializedSize(disServingRequestStats.kvCacheSize);
    return totalSize;
}

// ReqeuestStage

RequestStage Serialization::deserializeRequestStage(std::istream& is)
{
    int stage = su::deserialize<int>(is);
    return RequestStage{stage};
}

void Serialization::serialize(RequestStage const& requestStage, std::ostream& os)
{
    su::serialize(static_cast<int>(requestStage), os);
}

size_t Serialization::serializedSize(RequestStage const& requestStage)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(static_cast<int>(requestStage));
    return totalSize;
}

// RequestStats
RequestStats Serialization::deserializeRequestStats(std::istream& is)
{
    auto id = su::deserialize<IdType>(is);
    auto stage = su::deserialize<RequestStage>(is);
    auto contextPrefillPosition = su::deserialize<SizeType32>(is);
    auto numGeneratedTokens = su::deserialize<SizeType32>(is);
    auto avgNumDecodedTokensPerIter = su::deserialize<float>(is);
    auto scheduled = su::deserialize<bool>(is);
    auto paused = su::deserialize<bool>(is);
    auto disServingStats = su::deserialize<std::optional<DisServingRequestStats>>(is);
    auto allocTotalBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto allocNewBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto reusedBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto missedBlocksPerRequest = su::deserialize<SizeType32>(is);
    auto kvCacheHitRatePerRequest = su::deserialize<FloatType>(is);

    return RequestStats{id, stage, contextPrefillPosition, numGeneratedTokens, avgNumDecodedTokensPerIter, scheduled,
        paused, disServingStats, allocTotalBlocksPerRequest, allocNewBlocksPerRequest, reusedBlocksPerRequest,
        missedBlocksPerRequest, kvCacheHitRatePerRequest};
}

void Serialization::serialize(RequestStats const& state, std::ostream& os)
{
    su::serialize(state.id, os);
    su::serialize(state.stage, os);
    su::serialize(state.contextPrefillPosition, os);
    su::serialize(state.numGeneratedTokens, os);
    su::serialize(state.avgNumDecodedTokensPerIter, os);
    su::serialize(state.scheduled, os);
    su::serialize(state.paused, os);
    su::serialize(state.disServingStats, os);
    su::serialize(state.allocTotalBlocksPerRequest, os);
    su::serialize(state.allocNewBlocksPerRequest, os);
    su::serialize(state.reusedBlocksPerRequest, os);
    su::serialize(state.missedBlocksPerRequest, os);
    su::serialize(state.kvCacheHitRatePerRequest, os);
}

size_t Serialization::serializedSize(RequestStats const& state)
{

    size_t totalSize = 0;
    totalSize += su::serializedSize(state.id);
    totalSize += su::serializedSize((state.stage));
    totalSize += su::serializedSize(state.contextPrefillPosition);
    totalSize += su::serializedSize(state.numGeneratedTokens);
    totalSize += su::serializedSize(state.avgNumDecodedTokensPerIter);
    totalSize += su::serializedSize(state.scheduled);
    totalSize += su::serializedSize(state.paused);
    totalSize += su::serializedSize(state.disServingStats);
    totalSize += su::serializedSize(state.allocTotalBlocksPerRequest);
    totalSize += su::serializedSize(state.allocNewBlocksPerRequest);
    totalSize += su::serializedSize(state.reusedBlocksPerRequest);
    totalSize += su::serializedSize(state.missedBlocksPerRequest);
    totalSize += su::serializedSize(state.kvCacheHitRatePerRequest);
    return totalSize;
}

// RequestStatsPerIteration
RequestStatsPerIteration Serialization::deserializeRequestStatsPerIteration(std::istream& is)
{
    auto iter = su::deserialize<IterationType>(is);
    auto requestStats = su::deserialize<std::vector<RequestStats>>(is);

    return RequestStatsPerIteration{iter, requestStats};
}

RequestStatsPerIteration Serialization::deserializeRequestStatsPerIteration(std::vector<char>& buffer)
{
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);

    return Serialization::deserializeRequestStatsPerIteration(is);
}

void Serialization::serialize(RequestStatsPerIteration const& state, std::ostream& os)
{
    su::serialize(state.iter, os);
    su::serialize(state.requestStats, os);
}

std::vector<char> Serialization::serialize(RequestStatsPerIteration const& state)
{
    auto totalSize = Serialization::serializedSize(state);
    std::vector<char> buffer(totalSize);

    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    Serialization::serialize(state, os);

    return buffer;
}

size_t Serialization::serializedSize(RequestStatsPerIteration const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.iter);
    totalSize += su::serializedSize(state.requestStats);
    return totalSize;
}

std::vector<char> Serialization::serialize(std::vector<RequestStatsPerIteration> const& requestStatsVec)
{
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& requestStat : requestStatsVec)
    {
        totalSize += su::serializedSize(requestStat);
    }
    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);
    su::serialize(requestStatsVec.size(), os);
    for (auto const& requestStat : requestStatsVec)
    {
        su::serialize(requestStat, os);
    }
    return buffer;
}

std::vector<RequestStatsPerIteration> Serialization::deserializeRequestStatsPerIterationVec(std::vector<char>& buffer)
{

    std::vector<RequestStatsPerIteration> iterRequestStatsVec;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numIterStats = su::deserialize<std::size_t>(is);
    for (std::size_t iterStats = 0; iterStats < numIterStats; ++iterStats)
    {
        iterRequestStatsVec.emplace_back(Serialization::deserializeRequestStatsPerIteration(is));
    }
    return iterRequestStatsVec;
}

//  KVCacheEvents deque
std::vector<char> Serialization::serialize(std::deque<KVCacheEvent> const& eventQueue)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& event : eventQueue)
    {
        totalSize += su::serializedSize(event);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf(std::ios_base::out | std::ios_base::in);
    strbuf.pubsetbuf(buffer.data(), buffer.size());
    std::ostream os(&strbuf);

    su::serialize(eventQueue.size(), os);
    for (auto const& event : eventQueue)
    {
        su::serialize(event, os);
    }
    return buffer;
}

std::deque<KVCacheEvent> Serialization::deserializeKVCacheEvents(std::vector<char>& buffer)
{
    std::deque<KVCacheEvent> kvCacheEvents;
    su::VectorWrapBuf<char> strbuf(buffer);
    std::istream is(&strbuf);
    auto numEvents = su::deserialize<std::size_t>(is);
    for (std::size_t event = 0; event < numEvents; ++event)
    {
        kvCacheEvents.emplace_back(Serialization::deserializeKVCacheEvent(is));
    }
    return kvCacheEvents;
}

//  KVCacheEvent
size_t Serialization::serializedSize(KVCacheEvent const& event)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(event.eventId);
    totalSize += su::serializedSize(event.data);
    totalSize += su::serializedSize(event.windowSize);
    totalSize += su::serializedSize(event.attentionDpRank);
    return totalSize;
}

void Serialization::serialize(KVCacheEvent const& event, std::ostream& os)
{
    su::serialize(event.eventId, os);
    su::serialize(event.data, os);
    su::serialize(event.windowSize, os);
    su::serialize(event.attentionDpRank, os);
}

KVCacheEvent Serialization::deserializeKVCacheEvent(std::istream& is)
{
    auto eventId = su::deserialize<IdType>(is);
    auto data = su::deserialize<KVCacheEventData>(is);
    auto windowSize = su::deserialize<SizeType32>(is);
    auto attentionDpRank = su::deserialize<std::optional<SizeType32>>(is);

    return KVCacheEvent{eventId, data, windowSize, attentionDpRank};
}

// KVCacheCreatedData
size_t Serialization::serializedSize(KVCacheCreatedData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.numBlocksPerCacheLevel);
    return totalSize;
}

void Serialization::serialize(KVCacheCreatedData const& data, std::ostream& os)
{
    su::serialize(data.numBlocksPerCacheLevel, os);
}

KVCacheCreatedData Serialization::deserializeKVCacheCreatedData(std::istream& is)
{
    auto numBlocksPerCacheLevel = su::deserialize<std::vector<SizeType32>>(is);
    return KVCacheCreatedData{numBlocksPerCacheLevel};
}

// KVCacheStoredData
size_t Serialization::serializedSize(KVCacheStoredData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.parentHash);
    totalSize += su::serializedSize(data.blocks);
    return totalSize;
}

void Serialization::serialize(KVCacheStoredData const& data, std::ostream& os)
{
    su::serialize(data.parentHash, os);
    su::serialize(data.blocks, os);
}

KVCacheStoredData Serialization::deserializeKVCacheStoredData(std::istream& is)
{
    auto parentHash = su::deserialize<std::optional<IdType>>(is);
    auto blocks = su::deserialize<std::vector<KVCacheStoredBlockData>>(is);
    return KVCacheStoredData{parentHash, blocks};
}

// KVCacheStoredBlockData
size_t Serialization::serializedSize(KVCacheStoredBlockData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.blockHash);
    totalSize += su::serializedSize(data.tokens);
    totalSize += su::serializedSize(data.loraId);
    totalSize += su::serializedSize(data.cacheLevel);
    totalSize += su::serializedSize(data.priority);
    totalSize += su::serializedSize(data.mmKeys);
    return totalSize;
}

void Serialization::serialize(KVCacheStoredBlockData const& data, std::ostream& os)
{
    su::serialize(data.blockHash, os);
    su::serialize(data.tokens, os);
    su::serialize(data.loraId, os);
    su::serialize(data.cacheLevel, os);
    su::serialize(data.priority, os);
    su::serialize(data.mmKeys, os);
}

KVCacheStoredBlockData Serialization::deserializeKVCacheStoredBlockData(std::istream& is)
{
    auto blockHash = su::deserialize<IdType>(is);
    auto tokens = su::deserialize<tensorrt_llm::runtime::VecUniqueTokens>(is);
    auto loraId = su::deserialize<std::optional<tensorrt_llm::runtime::LoraTaskIdType>>(is);
    auto cacheLevel = su::deserialize<SizeType32>(is);
    auto priority = su::deserialize<SizeType32>(is);
    auto mmKeys = su::deserialize<std::vector<tensorrt_llm::batch_manager::kv_cache_manager::MmKey>>(is);

    return KVCacheStoredBlockData{blockHash, tokens, loraId, cacheLevel, priority, mmKeys};
}

// KVcacheRemovedData

size_t Serialization::serializedSize(KVCacheRemovedData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.blockHashes);
    return totalSize;
}

void Serialization::serialize(KVCacheRemovedData const& data, std::ostream& os)
{
    su::serialize(data.blockHashes, os);
}

KVCacheRemovedData Serialization::deserializeKVCacheRemovedData(std::istream& is)
{
    auto blockHashes = su::deserialize<std::vector<IdType>>(is);
    return KVCacheRemovedData{blockHashes};
}

// KVCacheEventDiff
template <typename T>
size_t Serialization::serializedSize(KVCacheEventDiff<T> const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.oldValue);
    totalSize += su::serializedSize(data.newValue);
    return totalSize;
}

template <typename T>
void Serialization::serialize(KVCacheEventDiff<T> const& data, std::ostream& os)
{
    su::serialize(data.oldValue, os);
    su::serialize(data.newValue, os);
}

template <typename T>
KVCacheEventDiff<T> Serialization::deserializeKVCacheEventDiff(std::istream& is)
{
    auto oldValue = su::deserialize<T>(is);
    auto newValue = su::deserialize<T>(is);
    return KVCacheEventDiff<T>{oldValue, newValue};
}

// KVCacheUpdatedData
size_t Serialization::serializedSize(KVCacheUpdatedData const& data)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(data.blockHash);
    totalSize += su::serializedSize(data.cacheLevel);
    totalSize += su::serializedSize(data.priority);
    return totalSize;
}

void Serialization::serialize(KVCacheUpdatedData const& data, std::ostream& os)
{
    su::serialize(data.blockHash, os);
    su::serialize(data.cacheLevel, os);
    su::serialize(data.priority, os);
}

KVCacheUpdatedData Serialization::deserializeKVCacheUpdatedData(std::istream& is)
{
    auto blockHash = su::deserialize<IdType>(is);
    auto cacheLevel = su::deserialize<std::optional<KVCacheEventDiff<SizeType32>>>(is);
    auto priority = su::deserialize<std::optional<KVCacheEventDiff<SizeType32>>>(is);
    return KVCacheUpdatedData{blockHash, cacheLevel, priority};
}

// MmKey
size_t Serialization::serializedSize(MmKey const& key)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(key.hash);
    totalSize += su::serializedSize(key.startOffset);
    totalSize += su::serializedSize(key.uuid);
    return totalSize;
}

void Serialization::serialize(MmKey const& key, std::ostream& os)
{
    su::serialize(key.hash, os);
    su::serialize(key.startOffset, os);
    su::serialize(key.uuid, os);
}

MmKey Serialization::deserializeMmKey(std::istream& is)
{
    auto hash = su::deserialize<std::array<uint8_t, 32>>(is);
    auto startOffset = su::deserialize<SizeType32>(is);
    auto uuid = su::deserialize<std::optional<std::string>>(is);
    return MmKey{std::move(hash), startOffset, std::move(uuid)};
}

// UniqueToken
size_t Serialization::serializedSize(tensorrt_llm::runtime::UniqueToken const& token)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(token.tokenId);
    totalSize += su::serializedSize(token.tokenExtraId);
    return totalSize;
}

void Serialization::serialize(tensorrt_llm::runtime::UniqueToken const& token, std::ostream& os)
{
    su::serialize(token.tokenId, os);
    su::serialize(token.tokenExtraId, os);
}

tensorrt_llm::runtime::UniqueToken Serialization::deserializeUniqueToken(std::istream& is)
{
    auto tokenId = su::deserialize<tensorrt_llm::runtime::TokenIdType>(is);
    auto tokenExtraId = su::deserialize<tensorrt_llm::runtime::TokenExtraIdType>(is);
    return tensorrt_llm::runtime::UniqueToken{tokenId, tokenExtraId};
}

// String
std::string Serialization::deserializeString(std::istream& is)
{
    return su::deserialize<std::string>(is);
}

// Bool
bool Serialization::deserializeBool(std::istream& is)
{
    return su::deserialize<bool>(is);
}

// ModelType
ModelType Serialization::deserializeModelType(std::istream& is)
{
    return su::deserialize<ModelType>(is);
}

// BlockKey (KV cache)
size_t Serialization::serializedSize(tensorrt_llm::batch_manager::kv_cache_manager::BlockKey const& key)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(key.usesExtraIds);
    totalSize += su::serializedSize(key.loraTaskId);
    totalSize += su::serializedSize(key.uniqueTokens);
    // std::vector<MmKey> where MmKey is pair<std::array<uint8_t,32>, SizeType32>
    totalSize += su::serializedSize(key.extraKeys);
    totalSize += su::serializedSize(key.cacheSaltID);
    return totalSize;
}

void Serialization::serialize(tensorrt_llm::batch_manager::kv_cache_manager::BlockKey const& key, std::ostream& os)
{
    su::serialize(key.usesExtraIds, os);
    su::serialize(key.loraTaskId, os);
    su::serialize(key.uniqueTokens, os);
    su::serialize(key.extraKeys, os);
    su::serialize(key.cacheSaltID, os);
}

tensorrt_llm::batch_manager::kv_cache_manager::BlockKey Serialization::deserializeBlockKey(std::istream& is)
{
    auto usesExtraIds = su::deserialize<bool>(is);
    auto loraTaskId = su::deserialize<std::optional<tensorrt_llm::batch_manager::kv_cache_manager::LoraTaskIdType>>(is);
    auto uniqueTokens = su::deserialize<std::vector<tensorrt_llm::runtime::UniqueToken>>(is);
    auto extraKeys = su::deserialize<std::vector<tensorrt_llm::batch_manager::kv_cache_manager::MmKey>>(is);
    auto cacheSaltID = su::deserialize<std::optional<CacheSaltIDType>>(is);
    tensorrt_llm::batch_manager::kv_cache_manager::BlockKey key;
    key.usesExtraIds = usesExtraIds;
    key.loraTaskId = std::move(loraTaskId);
    key.uniqueTokens = std::move(uniqueTokens);
    key.extraKeys = std::move(extraKeys);
    key.cacheSaltID = std::move(cacheSaltID);
    return key;
}

} // namespace tensorrt_llm::executor
""",
    'key-files/cpp/tensorrt_llm/kernels/fusedCatFp4.cu': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fusedCatFp4.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Constants (fixed for DSV3.2 FP4 indexer: head_dim=128, per-block-32 quant).
//
// INV_FP4_E2M1_MAX / MIN_AMAX mirror DeepGEMM's per_token_cast_to_fp4 reference
// (use_ue8m0=True, gran_k=32, use_packed_ue8m0=True) at
// tensorrt_llm/deep_gemm/utils/math.py. Keep in sync with that helper when
// bumping DeepGEMM — test_fused_cat_fp4_matches_deepgemm is the guard.
constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int ELEMS_PER_THREAD = 4;                           // 128 / 32 = 4 elements per thread.
constexpr int ROWS_PER_BLOCK = 8;                             // 8 rows per CTA (mirrors fusedCatFp8).
constexpr int FP4_BLOCK_SIZE = 32;                            // One UE8M0 scale per 32 elements.
constexpr int LANES_PER_FP4_BLOCK = 8;                        // 32 elements / 4 per lane = 8 lanes.
constexpr int FP4_BLOCKS_PER_ROW = HEAD_DIM / FP4_BLOCK_SIZE; // = 4.
constexpr float INV_FP4_E2M1_MAX = 1.0f / 6.0f;               // 1 / max representable FP4 E2M1 magnitude (6.0).
constexpr float MIN_AMAX = 1.0e-12f;                          // Floor on amax to keep ratio normal in ceil_to_ue8m0.

/// Helper union for vectorized BF16 loads (4 BF16 values = 8 bytes).
union BF16x4
{
    int2 vec;
    __nv_bfloat162 bf16x2[2];
};

/// FP4 E2M1 quantize a single scaled value.
/// Returns a 4-bit code matching DeepGEMM's table:
///   magnitudes {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
///   sign bit 3 set iff value < 0 and magnitude code != 0.
__device__ __forceinline__ uint32_t quantizeFp4E2M1(float scaled)
{
    float ax = fminf(fabsf(scaled), 6.0f);
    // Strict `>` matches Triton reference and DeepGEMM (boundary values round down).
    uint32_t idx = static_cast<uint32_t>(
        (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f));
    uint32_t code = idx & 0x7u;
    uint32_t sign = (scaled < 0.0f && idx != 0u) ? 1u : 0u;
    return code | (sign << 3);
}

/// Fused kernel: cat + per-block-32 FP4 E2M1 quantization (UE8M0 scales).
///
/// Grid: (ceil(M / ROWS_PER_BLOCK),)
/// Block: (WARP_SIZE * ROWS_PER_BLOCK,)   i.e., (256,)
///
/// Thread layout within a warp (handles one 128-element row):
///   - Thread t handles elements [4t, 4t+1, 4t+2, 4t+3].
///   - FP4 block g = t / 8 (group of 8 lanes); lane-in-group = t % 8.
///   - Per-block amax reduces across the 8 lanes of a group via shuffle (offsets 1, 2, 4).
///   - UE8M0 scale bit-trick runs on every lane; all 8 lanes in a group see
///     identical amax and thus produce the same scale.
///   - Output bytes: lane t writes packed[row*64 + 2t .. +1].
///   - Output scale: lane 0 gathers the 4 UE8M0 exponent bytes from lanes
///     {0, 8, 16, 24} and writes one int32.
__global__ __launch_bounds__(WARP_SIZE* ROWS_PER_BLOCK) void fusedCatFp4Kernel(int8_t* __restrict__ packed_out,
    int32_t* __restrict__ scale_out, __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t pe_row_stride, int32_t nope_row_stride)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    if (row >= M)
    {
        return;
    }

    // ---- Stage 1: Load + Concat (vectorized 8-byte loads) ----
    // pe_dim is guaranteed a multiple of ELEMS_PER_THREAD by the host check,
    // so each thread's 4 elements come entirely from pe or entirely from nope.
    float v0, v1, v2, v3;
    {
        int base = lane * ELEMS_PER_THREAD;
        __nv_bfloat16 const* pe_row = pe + static_cast<int64_t>(row) * pe_row_stride;
        __nv_bfloat16 const* nope_row = nope + static_cast<int64_t>(row) * nope_row_stride;

        bool from_pe = (base < pe_dim);
        __nv_bfloat16 const* src = from_pe ? pe_row : nope_row;
        int col = from_pe ? base : (base - pe_dim);

        BF16x4 loaded;
        loaded.vec = *reinterpret_cast<int2 const*>(src + col);

        float2 f0 = __bfloat1622float2(loaded.bf16x2[0]);
        float2 f1 = __bfloat1622float2(loaded.bf16x2[1]);
        v0 = f0.x;
        v1 = f0.y;
        v2 = f1.x;
        v3 = f1.y;
    }

    // ---- Stage 2: Per-block-32 amax reduction (within a group of 8 lanes) ----
    // __shfl_xor_sync with offsets 1, 2, 4 stays inside each group of 8 lanes,
    // so after three rounds all 8 lanes in a group share the same group-max.
    float local_max = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = local_max;
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
    amax = fmaxf(amax, MIN_AMAX);

    // ---- Stage 3: UE8M0 scale via IEEE 754 bit manipulation ----
    // scale = 2^ceil(log2(amax / 6)). With MIN_AMAX = 1e-12, ratio is always
    // normal (exp >= 1), so no explicit clamp needed.
    float ratio = amax * INV_FP4_E2M1_MAX;
    uint32_t bits = __float_as_uint(ratio);
    uint32_t exp_bits = bits & 0x7F800000u;
    if ((bits & 0x007FFFFFu) != 0u)
    {
        exp_bits += 0x00800000u;
    }
    float scale = __uint_as_float(exp_bits);

    // ---- Stage 4: FP4 E2M1 quantize ----
    // Use IEEE float division (compiles to div.rn.f32) for bit-exact parity
    // with DeepGEMM's `x / scale` reference. Reciprocal approximation could
    // drift at exact midpoints even though the relative error is below the
    // FP4 resolution on average.
    uint32_t c0 = quantizeFp4E2M1(v0 / scale);
    uint32_t c1 = quantizeFp4E2M1(v1 / scale);
    uint32_t c2 = quantizeFp4E2M1(v2 / scale);
    uint32_t c3 = quantizeFp4E2M1(v3 / scale);

    // ---- Stage 5: Nibble pack (even → low, odd → high) ----
    uint8_t byte0 = static_cast<uint8_t>(c0 | (c1 << 4));
    uint8_t byte1 = static_cast<uint8_t>(c2 | (c3 << 4));

    // ---- Stage 6: Store packed bytes (2 bytes per lane) ----
    int base_out = row * (HEAD_DIM / 2) + lane * 2;
    packed_out[base_out + 0] = static_cast<int8_t>(byte0);
    packed_out[base_out + 1] = static_cast<int8_t>(byte1);

    // ---- Stage 7: Pack the 4 UE8M0 exponent bytes into one int32 ----
    // All 8 lanes in a group share the same scale, so every lane has a valid
    // exp byte for its block. Shuffle the leader-lane values (0, 8, 16, 24)
    // to lane 0 and combine little-endian.
    uint32_t my_exp = (__float_as_uint(scale) >> 23) & 0xFFu;
    uint32_t b0 = __shfl_sync(0xFFFFFFFFu, my_exp, 0);
    uint32_t b1 = __shfl_sync(0xFFFFFFFFu, my_exp, 8);
    uint32_t b2 = __shfl_sync(0xFFFFFFFFu, my_exp, 16);
    uint32_t b3 = __shfl_sync(0xFFFFFFFFu, my_exp, 24);
    if (lane == 0)
    {
        uint32_t packed_scale = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        scale_out[row] = static_cast<int32_t>(packed_scale);
    }
}

} // anonymous namespace

void invokeFusedCatFp4(int8_t* packed_out, int32_t* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride,
    cudaStream_t stream)
{
    if (M == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(head_dim == HEAD_DIM, "fusedCatFp4: head_dim must be 128, got %d", head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim + nope_dim == head_dim, "fusedCatFp4: pe_dim (%d) + nope_dim (%d) != head_dim (%d)",
        pe_dim, nope_dim, head_dim);
    TLLM_CHECK_WITH_INFO(pe_dim % ELEMS_PER_THREAD == 0,
        "fusedCatFp4: pe_dim (%d) must be a multiple of %d for vectorized access", pe_dim, ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(
        pe_row_stride >= pe_dim, "fusedCatFp4: pe_row_stride (%d) must be >= pe_dim (%d)", pe_row_stride, pe_dim);
    TLLM_CHECK_WITH_INFO(nope_row_stride >= nope_dim, "fusedCatFp4: nope_row_stride (%d) must be >= nope_dim (%d)",
        nope_row_stride, nope_dim);
    TLLM_CHECK_WITH_INFO(pe_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp4: pe_row_stride (%d) must be a multiple of %d for aligned vectorized access", pe_row_stride,
        ELEMS_PER_THREAD);
    TLLM_CHECK_WITH_INFO(nope_row_stride % ELEMS_PER_THREAD == 0,
        "fusedCatFp4: nope_row_stride (%d) must be a multiple of %d for aligned vectorized access", nope_row_stride,
        ELEMS_PER_THREAD);

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARP_SIZE * ROWS_PER_BLOCK); // 256 threads per block

    fusedCatFp4Kernel<<<grid, block, 0, stream>>>(
        packed_out, scale_out, pe, nope, M, pe_dim, nope_dim, pe_row_stride, nope_row_stride);

    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
""",
    'key-files/cpp/tensorrt_llm/kernels/fusedCatFp4.h': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Fused concat + FP4 E2M1 per-block-32 quantization for the DSA indexer.
///
/// Given two BF16 input matrices `pe` [M, pe_dim] and `nope` [M, nope_dim],
/// this kernel concatenates them along the last dimension (pe first, nope second),
/// then quantizes each row to FP4 E2M1 with one UE8M0 scale per 32-element block
/// (head_dim=128 => 4 scales per row packed as one int32, little-endian).
///
/// Inputs need not be fully contiguous — only the innermost dimension must be
/// contiguous (stride 1). The row stride for each input is provided explicitly
/// via pe_row_stride / nope_row_stride, allowing processing non-contiguous
/// views (e.g. from torch.split()) without a prior contiguous copy.
///
/// Output packing (matches DeepGEMM `per_token_cast_to_fp4`, gran_k=32,
/// use_packed_ue8m0=True):
///   - packed[row, 2t + 0] = code[4t] | (code[4t+1] << 4)
///   - packed[row, 2t + 1] = code[4t+2] | (code[4t+3] << 4)
///   - scale[row] = ue8m0[block0] | (ue8m0[block1]<<8) | (ue8m0[block2]<<16) | (ue8m0[block3]<<24)
///
/// @param packed_out      Output packed FP4 bytes [M, head_dim/2], int8, row-major.
///                        Two E2M1 codes per byte (even-index in low nibble,
///                        odd-index in high nibble).
/// @param scale_out       Output UE8M0 scales [M, 1], int32. Each int32 packs
///                        four 8-bit exponents little-endian (block 0 → bits 0..7).
/// @param pe              Input PE part, BF16. Each row has pe_dim contiguous elements.
/// @param nope            Input non-PE part, BF16. Each row has nope_dim contiguous elements.
/// @param M               Number of rows (product of all dims except the last).
/// @param pe_dim          Dimension of PE input (must satisfy pe_dim + nope_dim == head_dim,
///                        and pe_dim must be a multiple of 4).
/// @param nope_dim        Dimension of non-PE input.
/// @param head_dim        Total head dimension (must be 128).
/// @param pe_row_stride   Stride (in elements) between consecutive rows of pe.
///                        For contiguous layout this equals pe_dim; for non-contiguous
///                        views (e.g. from torch.split) it may be larger.
/// @param nope_row_stride Stride (in elements) between consecutive rows of nope.
/// @param stream          CUDA stream.
void invokeFusedCatFp4(int8_t* packed_out, int32_t* scale_out, __nv_bfloat16 const* pe, __nv_bfloat16 const* nope,
    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride,
    cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
""",
    'key-files/cpp/tensorrt_llm/kernels/indexerKCacheGather.cu': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "IndexerKCacheGather.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
/**
 * Given a flat element index and tensor shape [d0, d1, d2, d3] with strides [s0, s1, s2, s3],
 * find the actual memory offset within the given k cache pool using the strides.
 */
__device__ __forceinline__ int64_t flatIndexToMemoryOffset(
    int64_t flat_idx, int32_t d0, int32_t d1, int32_t d2, int32_t d3, int64_t s0, int64_t s1, int64_t s2, int64_t s3)
{
    // Unravel from innermost to outermost dimension
    int32_t i3 = flat_idx % d3;
    flat_idx /= d3;

    int32_t i2 = flat_idx % d2;
    flat_idx /= d2;

    int32_t i1 = flat_idx % d1;
    flat_idx /= d1;

    int32_t i0 = flat_idx;

    // Compute memory offset using strides
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
}

} // anonymous namespace

/**
 * CUDA kernel to gather both FP8 K values and scales from the indexer k cache pool.
 * This is the inverse of indexerKCacheScatterUnifiedKernel.
 *
 * @param k_cache             Indexer k cache pool with shape [num_blocks, block_size, 1, per_token_size]
 *                            (can be non-contiguous)
 * @param slot_mapping_fp8    Flat element index for FP8 data start position [total_kv_len]
 * @param slot_mapping_scale  Flat element index for scale data start position [total_kv_len]
 * @param out_fp8             Output FP8 data [num_tokens, head_dim] contiguous
 * @param out_scale           Output scale data [num_tokens, scale_size] contiguous
 * @param k_token_start       Start offset into slot_mapping arrays
 * @param num_tokens          Number of tokens to gather
 * @param head_dim            Head dimension (must be 128)
 * @param scale_size          Scale size in bytes (must be 4)
 * @param cache_stride_0      Stride for k_cache dimension 0 (in bytes)
 * @param cache_stride_1      Stride for k_cache dimension 1 (in bytes)
 * @param cache_stride_2      Stride for k_cache dimension 2 (in bytes)
 * @param cache_stride_3      Stride for k_cache dimension 3 (in bytes)
 * @param cache_dim_0         Size of k_cache dimension 0
 * @param cache_dim_1         Size of k_cache dimension 1
 * @param cache_dim_2         Size of k_cache dimension 2
 * @param cache_dim_3         Size of k_cache dimension 3
 */
__global__ void indexerKCacheGatherUnifiedKernel(uint8_t const* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping_fp8, int64_t const* __restrict__ slot_mapping_scale,
    uint8_t* __restrict__ out_fp8, uint8_t* __restrict__ out_scale, int32_t k_token_start, int32_t num_tokens,
    int32_t head_dim, int32_t scale_size, int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2,
    int64_t cache_stride_3, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3)
{
    // For head_dim=128, each thread handles 4 bytes/elements per read/write instruction
    constexpr int VEC_SIZE = 4;

    // Token index from block.x
    int32_t token_idx = blockIdx.x;

    if (token_idx >= num_tokens)
    {
        return;
    }

    // Index into slot_mapping with k_token_start offset
    int32_t slot_idx = k_token_start + token_idx;

    int64_t flat_idx_fp8_base = slot_mapping_fp8[slot_idx];
    int64_t flat_idx_scale_base = slot_mapping_scale[slot_idx];

    if (flat_idx_fp8_base < 0 || flat_idx_scale_base < 0)
    {
        return;
    }

    int32_t head_dim_idx = threadIdx.x * VEC_SIZE;
    int64_t flat_idx = flat_idx_fp8_base + head_dim_idx;

    // Convert flat index to memory offset using strides (k cache pool from cpp kv cache manager is non-contiguous)
    int64_t src_offset = flatIndexToMemoryOffset(flat_idx, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3,
        cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3);
    int64_t dst_offset = token_idx * head_dim + head_dim_idx;

    // 4 bytes read from non-contiguous cache, write to contiguous output
    *reinterpret_cast<uint32_t*>(&out_fp8[dst_offset]) = *reinterpret_cast<uint32_t const*>(&k_cache[src_offset]);

    // Only thread 0 reads the single 4 bytes scale value
    if (threadIdx.x == 0)
    {
        int64_t src_offset_scale = flatIndexToMemoryOffset(flat_idx_scale_base, cache_dim_0, cache_dim_1, cache_dim_2,
            cache_dim_3, cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3);
        int64_t dst_offset_scale = token_idx * scale_size; // scale_size = 4

        // 4 bytes read for scale
        *reinterpret_cast<uint32_t*>(&out_scale[dst_offset_scale])
            = *reinterpret_cast<uint32_t const*>(&k_cache[src_offset_scale]);
    }
}

void invokeIndexerKCacheGather(uint8_t const* k_cache, int64_t const* slot_mapping_fp8,
    int64_t const* slot_mapping_scale, uint8_t* out_fp8, uint8_t* out_scale, int32_t k_token_start, int32_t num_tokens,
    int32_t head_dim, int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2,
    int32_t cache_dim_3, int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3,
    cudaStream_t stream)
{
    if (num_tokens == 0)
    {
        return;
    }

    // Each thread reads 4 bytes; head_dim is the byte-width of one token in
    // the cache (128 for FP8, 64 for FP4 packed). Scale is a single 4-byte word.
    constexpr int32_t VEC_SIZE = 4;
    TLLM_CHECK_WITH_INFO(head_dim == 128 || head_dim == 64,
        "head_dim must be 128 (FP8) or 64 (FP4 packed) for the indexer cache (got %d)", head_dim);
    TLLM_CHECK_WITH_INFO(head_dim % VEC_SIZE == 0, "head_dim (%d) must be a multiple of %d", head_dim, VEC_SIZE);
    TLLM_CHECK_WITH_INFO(scale_size == 4,
        "scale_size must equal 4 bytes (packed UE8M0 x4 for FP4, 1 float32 for FP8, got %d)", scale_size);

    int32_t const threads_per_block = head_dim / VEC_SIZE;

    dim3 block(threads_per_block);
    dim3 grid(num_tokens);

    indexerKCacheGatherUnifiedKernel<<<grid, block, 0, stream>>>(k_cache, slot_mapping_fp8, slot_mapping_scale, out_fp8,
        out_scale, k_token_start, num_tokens, head_dim, scale_size, cache_stride_0, cache_stride_1, cache_stride_2,
        cache_stride_3, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3);

    // Check for kernel launch errors
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
""",
    'key-files/cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "IndexerKCacheScatter.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
/**
 * Given a flat element index and tensor shape [d0, d1, d2, d3] with strides [s0, s1, s2, s3],
 * find the actual memory offset within the given k cache pool using the strides.
 */
__device__ __forceinline__ int64_t flatIndexToMemoryOffset(
    int64_t flat_idx, int32_t d0, int32_t d1, int32_t d2, int32_t d3, int64_t s0, int64_t s1, int64_t s2, int64_t s3)
{
    // Unravel from innermost to outermost dimension
    int32_t i3 = flat_idx % d3;
    flat_idx /= d3;

    int32_t i2 = flat_idx % d2;
    flat_idx /= d2;

    int32_t i1 = flat_idx % d1;
    flat_idx /= d1;

    int32_t i0 = flat_idx;

    // Compute memory offset using strides
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
}

} // anonymous namespace

/**
 * CUDA kernel to scatter both FP8 K values and scales into the indexer k cache pool
 *
 * @param k_fp8_bytes       Quantized FP8 data [num_tokens, 128]
 * @param k_scale_bytes     Quantized scales (1 per token) [num_tokens, 4]
 * @param k_cache           Indexer k cache pool with shape [num_blocks, block_size, 1, per_token_size] (can be
 * non-contiguous)
 * @param slot_mapping_fp8  Flat element index for FP8 data start position [num_tokens]
 * @param slot_mapping_scale Flat element index for scale data start position [num_tokens]
 * @param num_tokens        Number of tokens
 * @param head_dim          Head dimension (must be 128)
 * @param scale_size        Scale size in bytes (must be 4)
 * @param cache_stride_0    Stride for k_cache dimension 0 (in bytes)
 * @param cache_stride_1    Stride for k_cache dimension 1 (in bytes)
 * @param cache_stride_2    Stride for k_cache dimension 2 (in bytes)
 * @param cache_stride_3    Stride for k_cache dimension 3 (in bytes)
 * @param cache_dim_0       Size of k_cache dimension 0
 * @param cache_dim_1       Size of k_cache dimension 1
 * @param cache_dim_2       Size of k_cache dimension 2
 * @param cache_dim_3       Size of k_cache dimension 3
 */
__global__ void indexerKCacheScatterUnifiedKernel(uint8_t const* __restrict__ k_fp8_bytes,
    uint8_t const* __restrict__ k_scale_bytes, uint8_t* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping_fp8, int64_t const* __restrict__ slot_mapping_scale, int32_t num_tokens,
    int32_t head_dim, int32_t scale_size, int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2,
    int64_t cache_stride_3, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3)
{
    // For head_dim=128, each thread handles 4 bytes/elements per read/write instruction
    constexpr int VEC_SIZE = 4;

    // Token index from block.x
    int32_t token_idx = blockIdx.x;

    if (token_idx >= num_tokens)
    {
        return;
    }

    int64_t flat_idx_fp8_base = slot_mapping_fp8[token_idx];
    int64_t flat_idx_scale_base = slot_mapping_scale[token_idx];

    if (flat_idx_fp8_base < 0 || flat_idx_scale_base < 0)
    {
        return;
    }

    int32_t head_dim_idx = threadIdx.x * VEC_SIZE;
    int64_t flat_idx = flat_idx_fp8_base + head_dim_idx;

    // Convert flat index to memory offset using strides (k cache pool from cpp kv cache manager is non-contiguous)
    int64_t dst_offset = flatIndexToMemoryOffset(flat_idx, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3,
        cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3);
    int64_t src_offset = token_idx * head_dim + head_dim_idx;

    // 4 bytes write
    *reinterpret_cast<uint32_t*>(&k_cache[dst_offset]) = *reinterpret_cast<uint32_t const*>(&k_fp8_bytes[src_offset]);

    // Only thread 0 writes the single 4 bytes scale value
    if (threadIdx.x == 0)
    {
        int64_t dst_offset_scale = flatIndexToMemoryOffset(flat_idx_scale_base, cache_dim_0, cache_dim_1, cache_dim_2,
            cache_dim_3, cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3);
        int64_t src_offset_scale = token_idx * scale_size; // scale_size = 4

        // 4 bytes write for scale
        *reinterpret_cast<uint32_t*>(&k_cache[dst_offset_scale])
            = *reinterpret_cast<uint32_t const*>(&k_scale_bytes[src_offset_scale]);
    }
}

void invokeIndexerKCacheScatter(uint8_t const* k_fp8_bytes, uint8_t const* k_scale_bytes, uint8_t* k_cache,
    int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t num_tokens, int32_t head_dim,
    int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3,
    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3, cudaStream_t stream)
{
    if (num_tokens == 0)
    {
        return;
    }

    // Each thread writes 4 bytes (VEC_SIZE in the kernel). head_dim is the
    // number of bytes per token (128 for FP8, 64 for FP4 packed). Scale is a
    // single 4-byte word per token, written by thread 0.
    constexpr int32_t VEC_SIZE = 4;
    TLLM_CHECK_WITH_INFO(head_dim == 128 || head_dim == 64,
        "head_dim must be 128 (FP8) or 64 (FP4 packed) for the indexer cache (got %d)", head_dim);
    TLLM_CHECK_WITH_INFO(head_dim % VEC_SIZE == 0, "head_dim (%d) must be a multiple of %d", head_dim, VEC_SIZE);
    TLLM_CHECK_WITH_INFO(scale_size == 4,
        "scale_size must equal 4 bytes (packed UE8M0 x4 for FP4, 1 float32 for FP8, got %d)", scale_size);

    int32_t const threads_per_block = head_dim / VEC_SIZE;

    dim3 block(threads_per_block);
    dim3 grid(num_tokens);

    indexerKCacheScatterUnifiedKernel<<<grid, block, 0, stream>>>(k_fp8_bytes, k_scale_bytes, k_cache, slot_mapping_fp8,
        slot_mapping_scale, num_tokens, head_dim, scale_size, cache_stride_0, cache_stride_1, cache_stride_2,
        cache_stride_3, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3);

    // Check for kernel launch errors
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
""",
    'key-files/cpp/tensorrt_llm/nanobind/batch_manager/kvCacheManager.cpp': r"""/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/nanobind/common/bindTypes.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>

namespace tb = tensorrt_llm::batch_manager;
namespace tbc = tensorrt_llm::batch_manager::kv_connector;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tr = tensorrt_llm::runtime;
namespace nb = nanobind;
using BlockKey = tbk::BlockKey;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
using CacheBlockIds = std::vector<std::vector<SizeType32>>;

NB_MAKE_OPAQUE(CacheBlockIds);

namespace
{
std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

class PyKvCacheManager : public tbk::BaseKVCacheManager
{
public:
    NB_TRAMPOLINE(tbk::BaseKVCacheManager, 39);

    // using BaseKVCacheManager::BaseKVCacheManager; // Inherit constructors
    void allocatePools(bool useUvm = false) override
    {
        NB_OVERRIDE_PURE(allocatePools, useUvm);
    }

    void releasePools() override
    {
        NB_OVERRIDE_PURE(releasePools);
    }

    void startScheduling() override
    {
        NB_OVERRIDE_PURE(startScheduling);
    }

    SizeType32 getTokensPerBlock() const override
    {
        NB_OVERRIDE_PURE(getTokensPerBlock);
    }

    SizeType32 getMaxNumBlocks() const override
    {
        NB_OVERRIDE_PURE(getMaxNumBlocks);
    }

    SizeType32 getNumPools() const override
    {
        NB_OVERRIDE_PURE(getNumPools);
    }

    tbk::KvCacheStats getKvCacheStats() const override
    {
        NB_OVERRIDE_PURE(getKvCacheStats);
    }

    std::map<SizeType32, tbk::KvCacheIterationStats> getIterationStats() override
    {
        NB_OVERRIDE_PURE(getIterationStats);
    }

    void addToken(tb::LlmRequest::RequestIdType requestId) override
    {
        NB_OVERRIDE_PURE(addToken, requestId);
    }

    SizeType32 getTokenCount(tb::LlmRequest::RequestIdType requestId) const override
    {
        NB_OVERRIDE_PURE(getTokenCount, requestId);
    }

    void addSequenceBatch(
        std::vector<std::tuple<tb::LlmRequest::RequestIdType, SizeType32, SizeType32>> const& requestInfos,
        std::vector<std::reference_wrapper<tb::LlmRequest>> const& llmRequests) override
    {
        NB_OVERRIDE_PURE(addSequenceBatch, requestInfos, llmRequests);
    }

    std::optional<tbk::KVCacheBlock::IdType> removeSequence(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest = std::nullopt,
        bool pinOnRelease = false) override
    {
        NB_OVERRIDE_PURE(removeSequence, requestId, llmRequest, pinOnRelease);
    }

    std::vector<tbk::KVCacheBlock::IdType> storeBlocksForReuse(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest, bool pinBlocks) override
    {
        NB_OVERRIDE_PURE(storeBlocksForReuse, requestId, llmRequest, pinBlocks);
    }

    tbk::GenerationRequest const& getSequence(tb::LlmRequest::RequestIdType requestId) const override
    {
        NB_OVERRIDE_PURE(getSequence, requestId);
    }

    void schedulingRemoveSequence(tb::LlmRequest::RequestIdType requestId) override
    {
        NB_OVERRIDE_PURE(schedulingRemoveSequence, requestId);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        NB_OVERRIDE_PURE(getBlockPoolPointers);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        NB_OVERRIDE_PURE(getLayerToPoolMapping);
    }

    void getBlockOffsetsOfBatch(tensorrt_llm::runtime::ITensor& output, SizeType32 firstBatchSlotIdx,
        SizeType32 batchSize, SizeType32 beamWidth) const override
    {
        NB_OVERRIDE_PURE(getBlockOffsetsOfBatch, output, firstBatchSlotIdx, batchSize, beamWidth);
    }

    SizeType32 copyBlockOffsets(tensorrt_llm::runtime::ITensor& output, SizeType32 outputSlotOffset,
        tb::LlmRequest::RequestIdType requestId) const override
    {
        NB_OVERRIDE_PURE(copyBlockOffsets, output, outputSlotOffset, requestId);
    }

    bool isEnableBlockReuse() const override
    {
        NB_OVERRIDE_PURE(isEnableBlockReuse);
    }

    tbk::PrefixReuseSummary analyzePrefixReuse(
        tbk::VecUniqueTokens const& uniqueTokens, tb::LlmRequest const& llmRequest) const override
    {
        NB_OVERRIDE_PURE(analyzePrefixReuse, uniqueTokens, llmRequest);
    }

    bool isEnablePartialReuse() const override
    {
        NB_OVERRIDE_PURE(isEnablePartialReuse);
    }

    void rewindKVCache(tb::LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override
    {
        NB_OVERRIDE_PURE(rewindKVCache, requestId, rewindLengths);
    }

    bool isCrossKv() const override
    {
        NB_OVERRIDE_PURE(isCrossKv);
    }

    void storeContextBlocks(tb::LlmRequest const& llmRequest) override
    {
        NB_OVERRIDE_PURE(storeContextBlocks, llmRequest);
    }

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        tb::LlmRequest::RequestIdType requestId, SizeType32 windowSize) const override
    {
        NB_OVERRIDE_PURE(getCacheBlockIds, requestId, windowSize);
    }

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const override
    {
        NB_OVERRIDE_PURE(getBatchCacheBlockIds, requestIds, windowSize);
    }

    SizeType32 getUsedNumBlocks() const override
    {
        NB_OVERRIDE_PURE(getUsedNumBlocks);
    }

    SizeType32 getNumFreeBlocks() const override
    {
        NB_OVERRIDE_PURE(getNumFreeBlocks);
    }

    tbk::BlockManager const& getBlockManager() const override
    {
        NB_OVERRIDE_PURE(getBlockManager);
    }

    std::deque<tensorrt_llm::executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        NB_OVERRIDE_PURE(getLatestEvents, timeout);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override
    {
        NB_OVERRIDE_PURE(getPrimaryPool, layer_idx);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getIndexerKCachePool() const override
    {
        NB_OVERRIDE_PURE(getIndexerKCachePool);
    }

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        NB_OVERRIDE_PURE(getPoolLayerIdx, layer_idx);
    }

    void syncTransferManagerWithBufferManager() override
    {
        NB_OVERRIDE_PURE(syncTransferManagerWithBufferManager);
    }

    void refreshBlocks() override
    {
        NB_OVERRIDE_PURE(refreshBlocks);
    }

    void flushIterationEvents() override
    {
        NB_OVERRIDE_PURE(flushIterationEvents);
    }
};

// TODO: Deduplicate executor bindings KvCacheStats
class PyBasePeftCacheManager : public tb::BasePeftCacheManager
{
public:
    ~PyBasePeftCacheManager() override = default;

    NB_TRAMPOLINE(tb::BasePeftCacheManager, 8);

    void addRequestPeft(tb::BasePeftCacheManager::LlmRequestPtr llmRequest, bool tryGpuCache = true) override
    {
        NB_OVERRIDE_PURE(addRequestPeft, llmRequest, tryGpuCache);
    }

    tb::BasePeftCacheManager::PeftTable ensureBatch(tb::RequestVector const& contextRequests,
        tb::RequestVector const& generationRequests, bool resetGpuCache = false) override
    {
        NB_OVERRIDE_PURE(ensureBatch, contextRequests, generationRequests, resetGpuCache);
    }

    void resetDeviceCache() override
    {
        NB_OVERRIDE_PURE(resetDeviceCache);
    }

    void markRequestDone(tb::LlmRequest const& llmReq, bool pause = false) override
    {
        NB_OVERRIDE_PURE(markRequestDone, llmReq, pause);
    }

    tr::SizeType32 getMaxDevicePages() const override
    {
        NB_OVERRIDE_PURE(getMaxDevicePages);
    }

    tr::SizeType32 getMaxHostPages() const override
    {
        NB_OVERRIDE_PURE(getMaxHostPages);
    }

    tr::SizeType32 determineNumPages(std::shared_ptr<tb::LlmRequest> llmRequest) const override
    {
        NB_OVERRIDE_PURE(determineNumPages, llmRequest);
    }

    bool enabled() const override
    {
        NB_OVERRIDE_PURE(enabled);
    }
};
} // namespace

void tb::kv_cache_manager::KVCacheManagerBindings::initBindings(nb::module_& m)
{
    nb::class_<tbk::LinearAttentionMetadata>(m, "LinearAttentionMetadata")
        .def(nb::init<>())
        .def_rw("linear_layer_indices", &tbk::LinearAttentionMetadata::linearLayerIndices)
        .def_rw("cache_type", &tbk::LinearAttentionMetadata::cacheType)
        .def_rw("all_recurrent_states_bytes", &tbk::LinearAttentionMetadata::allRecurrentStatesBytes)
        .def_rw("states_snapshot_interval", &tbk::LinearAttentionMetadata::statesSnapshotInterval)
        .def_rw("save_last_snapshot", &tbk::LinearAttentionMetadata::saveLastSnapshot);

    nb::enum_<tbk::LinearAttentionMetadata::LinearCacheType>(m, "LinearCacheType")
        .value("RECURRENT_STATES", tbk::LinearAttentionMetadata::LinearCacheType::kRecurrentStates);

    nb::class_<tbk::PrefixReuseSummary>(m, "PrefixReuseSummary")
        .def(nb::init<>())
        .def_ro("reusable_blocks_allocated", &tbk::PrefixReuseSummary::reusableBlocksAllocated)
        .def_ro("reusable_blocks_all", &tbk::PrefixReuseSummary::reusableBlocksAll)
        .def_ro("first_new_block", &tbk::PrefixReuseSummary::firstNewBlock);

    nb::class_<tbk::KvCacheStats>(m, "KvCacheStats")
        .def(nb::init<>())
        .def_rw("max_num_blocks", &tbk::KvCacheStats::maxNumBlocks)
        .def_rw("free_num_blocks", &tbk::KvCacheStats::freeNumBlocks)
        .def_rw("used_num_blocks", &tbk::KvCacheStats::usedNumBlocks)
        .def_rw("tokens_per_block", &tbk::KvCacheStats::toksPerBlock)
        .def_rw("alloc_total_blocks", &tbk::KvCacheStats::allocTotalBlocks)
        .def_rw("alloc_new_blocks", &tbk::KvCacheStats::allocNewBlocks)
        .def_rw("reused_blocks", &tbk::KvCacheStats::reusedBlocks)
        .def_rw("missed_blocks", &tbk::KvCacheStats::missedBlocks)
        .def_rw("cache_hit_rate", &tbk::KvCacheStats::cacheHitRate)
        .def_rw("num_free_blocks_per_window_size", &tbk::KvCacheStats::numFreeBlocksPerWindowSize)
        .def_rw("allocated_bytes", &tbk::KvCacheStats::allocatedBytes);

    nb::class_<tbk::KvCacheIterationStats>(m, "KvCacheIterationStats")
        .def(nb::init<>())
        .def_rw("primary_max_num_blocks", &tbk::KvCacheIterationStats::primaryMaxNumBlocks)
        .def_rw("primary_free_num_blocks", &tbk::KvCacheIterationStats::primaryFreeNumBlocks)
        .def_rw("primary_used_num_blocks", &tbk::KvCacheIterationStats::primaryUsedNumBlocks)
        .def_rw("secondary_max_num_blocks", &tbk::KvCacheIterationStats::secondaryMaxNumBlocks)
        .def_rw("secondary_free_num_blocks", &tbk::KvCacheIterationStats::secondaryFreeNumBlocks)
        .def_rw("secondary_used_num_blocks", &tbk::KvCacheIterationStats::secondaryUsedNumBlocks)
        .def_rw("iter_alloc_total_blocks", &tbk::KvCacheIterationStats::iterAllocTotalBlocks)
        .def_rw("iter_alloc_new_blocks", &tbk::KvCacheIterationStats::iterAllocNewBlocks)
        .def_rw("iter_reused_blocks", &tbk::KvCacheIterationStats::iterReusedBlocks)
        .def_rw("iter_full_reused_blocks", &tbk::KvCacheIterationStats::iterFullReusedBlocks)
        .def_rw("iter_partial_reused_blocks", &tbk::KvCacheIterationStats::iterPartialReusedBlocks)
        .def_rw("iter_missed_blocks", &tbk::KvCacheIterationStats::iterMissedBlocks)
        .def_rw("iter_cache_hit_rate", &tbk::KvCacheIterationStats::iterCacheHitRate)
        .def_rw("iter_gen_alloc_blocks", &tbk::KvCacheIterationStats::iterGenAllocBlocks)
        .def_rw("iter_onboard_blocks", &tbk::KvCacheIterationStats::iterOnboardBlocks)
        .def_rw("iter_onboard_bytes", &tbk::KvCacheIterationStats::iterOnboardBytes)
        .def_rw("iter_offload_blocks", &tbk::KvCacheIterationStats::iterOffloadBlocks)
        .def_rw("iter_offload_bytes", &tbk::KvCacheIterationStats::iterOffloadBytes)
        .def_rw("iter_intra_device_copy_blocks", &tbk::KvCacheIterationStats::iterIntraDeviceCopyBlocks)
        .def_rw("iter_intra_device_copy_bytes", &tbk::KvCacheIterationStats::iterIntraDeviceCopyBytes);

    nb::class_<tbk::BlockKey>(m, "BlockKey")
        .def(nb::init<>())
        .def(nb::init<VecTokens const&, std::optional<tr::LoraTaskIdType>>(), nb::arg("tokens"),
            nb::arg("lora_task_id") = std::nullopt)
        .def(nb::init<bool, std::optional<tr::LoraTaskIdType>, VecUniqueTokens const&>(), nb::arg("uses_extra_ids"),
            nb::arg("lora_task_id"), nb::arg("unique_tokens"))
        .def_ro("uses_extra_ids", &tbk::BlockKey::usesExtraIds)
        .def_ro("lora_task_id", &tbk::BlockKey::loraTaskId)
        .def_ro("unique_tokens", &tbk::BlockKey::uniqueTokens)
        .def("__eq__", &tbk::BlockKey::operator==)
        .def("__hash__", [](tbk::BlockKey const& key) -> size_t { return tbk::BlockKeyHasher{}(key); });

    nb::class_<tbk::BlockKeyHasher>(m, "BlockKeyHasher")
        .def_static("hash", &tbk::BlockKeyHasher::hash, nb::arg("block_key"), nb::arg("parent_hash") = 0);

    nb::class_<tbk::KVCacheEventManager>(m, "KVCacheEventManager")
        .def(nb::init<size_t, std::optional<SizeType32>, std::optional<SizeType32>, SizeType32>(),
            nb::arg("max_kv_event_entries"), nb::arg("attention_dp_rank") = std::nullopt,
            nb::arg("attention_dp_size") = std::nullopt, nb::arg("attention_dp_events_gather_period_ms") = 5);

    nb::class_<tbk::BaseKVCacheManager, PyKvCacheManager>(m, "BaseKVCacheManager")
        .def_static("calculate_max_num_blocks", &tbk::BaseKVCacheManager::calculateMaxNumBlocks, nb::arg("config"),
            nb::arg("dtype"), nb::arg("num_kv_heads_per_layer"), nb::arg("size_per_head"), nb::arg("tokens_per_block"),
            nb::arg("world_config"), nb::arg("window_size_to_layers"), nb::arg("allotted_primary_mem_bytes"),
            nb::arg("allotted_secondary_mem_bytes"), nb::arg("extra_cost_memory"), nb::arg("kv_factor"),
            nb::arg("max_batch_size"), nb::arg("linear_attention_metadata") = std::nullopt,
            nb::call_guard<nb::gil_scoped_release>())
        .def("allocate_pools", &BaseKVCacheManager::allocatePools, nb::call_guard<nb::gil_scoped_release>())
        .def("release_pools", &BaseKVCacheManager::releasePools, nb::call_guard<nb::gil_scoped_release>())
        .def("start_scheduling", &BaseKVCacheManager::startScheduling, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("tokens_per_block", &BaseKVCacheManager::getTokensPerBlock)
        .def_prop_ro("max_num_blocks", &BaseKVCacheManager::getMaxNumBlocks)
        .def_prop_ro("num_pools", &BaseKVCacheManager::getNumPools)
        .def("get_kv_cache_stats", &BaseKVCacheManager::getKvCacheStats, nb::call_guard<nb::gil_scoped_release>())
        .def("get_iteration_stats", &BaseKVCacheManager::getIterationStats, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("max_blocks_per_seq",
            [](tbk::BaseKVCacheManager& self) { return self.getOffsetTableDimensions().maxBlocksPerSeq; })
        // nanobind does not inherit C++ default arguments from member-function pointers, so the
        // cached_summary default must be spelled out here or Python callers that omit it raise
        // TypeError. See kvCacheManager.h for the C++ signatures.
        .def("get_needed_blocks_one_step", &BaseKVCacheManager::getNeededBlocksOneStep, nb::arg("req"),
            nb::arg("two_steps_look_ahead"), nb::arg("window_size"), nb::arg("cached_summary") = std::nullopt,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_remaining_blocks_to_completion", &BaseKVCacheManager::getRemainingBlocksToCompletion, nb::arg("req"),
            nb::arg("window_size"), nb::arg("cached_summary") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def("add_token", &BaseKVCacheManager::addToken, nb::call_guard<nb::gil_scoped_release>())
        .def("get_token_count", &BaseKVCacheManager::getTokenCount, nb::arg("request_id"))
        .def(
            "add_sequence_batch",
            [](tbk::BaseKVCacheManager& self, nb::list requestInfosList, nb::list llmRequestsList)
            {
                // Marshal Python inputs while GIL is held.
                std::vector<std::tuple<tb::LlmRequest::RequestIdType, SizeType32, SizeType32>> requestInfos;
                std::vector<std::reference_wrapper<tb::LlmRequest>> llmRequests;
                requestInfos.reserve(nb::len(requestInfosList));
                llmRequests.reserve(nb::len(llmRequestsList));
                for (size_t i = 0; i < nb::len(requestInfosList); ++i)
                {
                    auto info = nb::cast<nb::tuple>(requestInfosList[i]);
                    requestInfos.emplace_back(nb::cast<tb::LlmRequest::RequestIdType>(info[0]),
                        nb::cast<SizeType32>(info[1]), nb::cast<SizeType32>(info[2]));
                    llmRequests.push_back(std::ref(nb::cast<tb::LlmRequest&>(llmRequestsList[i])));
                }
                // Release GIL only for the C++ call.
                nb::gil_scoped_release release;
                self.addSequenceBatch(requestInfos, llmRequests);
            },
            nb::arg("request_infos"), nb::arg("llm_requests"))
        .def("remove_sequence", &BaseKVCacheManager::removeSequence, nb::call_guard<nb::gil_scoped_release>())
        .def("pin_blocks", &BaseKVCacheManager::pinBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def("scheduling_remove_sequence", &BaseKVCacheManager::schedulingRemoveSequence,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_block_pool_pointers",
            [](tbk::BaseKVCacheManager& self)
            {
                std::optional<at::Tensor> block_pool_pointers{std::nullopt};
                auto tensor = self.getBlockPoolPointers();
                if (tensor)
                {
                    std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                    block_pool_pointers = tr::Torch::tensor(_tensor);
                }
                return block_pool_pointers;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_block_scale_pool_pointers",
            [](tbk::BaseKVCacheManager& self)
            {
                std::optional<at::Tensor> block_scale_pool_pointers{std::nullopt};
                auto tensor = self.getBlockScalePoolPointers();
                if (tensor)
                {
                    std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                    block_scale_pool_pointers = tr::Torch::tensor(_tensor);
                }
                return block_scale_pool_pointers;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_layer_to_pool_mapping",
            [](tbk::BaseKVCacheManager& self)
            {
                std::optional<at::Tensor> layer_to_pool_mapping{std::nullopt};
                auto tensor = self.getLayerToPoolMapping();
                if (tensor)
                {
                    std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                    layer_to_pool_mapping = tr::Torch::tensor(_tensor);
                }
                return layer_to_pool_mapping;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_primary_pool_data",
            [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
            {
                auto pool = tr::Torch::tensor(self.getPrimaryPool(layer_idx));
                auto pool_layer_idx = self.getPoolLayerIdx(layer_idx);
                if (self.isPoolLayerFirst(layer_idx))
                {
                    // Layer-first layout: pool[pool_layer_idx, :]
                    return pool.index({pool_layer_idx});
                }
                // Standard layout: pool[:, pool_layer_idx]
                return pool.index({torch::indexing::Slice(), pool_layer_idx});
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_recurrent_states_pool",
            [](tbk::BaseKVCacheManager& self) -> at::Tensor
            {
                auto const& pool = self.getBlockManager().getRecurrentStatesPool();
                return tr::Torch::tensor(pool.primaryPtr);
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_indexer_k_cache_pool_data",
            [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
            {
                auto pool = tr::Torch::tensor(self.getIndexerKCachePool());
                return pool.index({torch::indexing::Slice(), layer_idx});
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_indexer_k_cache_pool",
            [](tbk::BaseKVCacheManager& self) -> at::Tensor { return tr::Torch::tensor(self.getIndexerKCachePool()); },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_unique_primary_pool", [](tbk::BaseKVCacheManager& self) { return self.getUniquePrimaryPool(); },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_block_offsets_of_batch",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                SizeType32 beamWidth)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                self.getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "copy_block_offsets",
            [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 outputSlotOffset,
                tb::LlmRequest::RequestIdType requestId)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                auto maxBlockCount = self.copyBlockOffsets(*(_output.value()), outputSlotOffset, requestId);
                return maxBlockCount;
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "copy_batch_block_offsets",
            [](tbk::BaseKVCacheManager& self, at::Tensor output,
                std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 const beamWidth,
                SizeType32 const offset)
            {
                auto _output = from_torch(output);
                TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                for (size_t i = 0; i < requestIds.size(); ++i)
                {
                    self.copyBlockOffsets(*(_output.value()), i * beamWidth + offset, requestIds[i]);
                }
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_latest_events",
            [](tbk::BaseKVCacheManager& self, std::optional<double> timeout_ms = std::nullopt)
            {
                if (timeout_ms)
                {
                    return self.getLatestEvents(std::chrono::milliseconds(static_cast<int64_t>(*timeout_ms)));
                }
                return self.getLatestEvents(std::nullopt);
            },
            nb::arg("timeout_ms") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("enable_block_reuse", &BaseKVCacheManager::isEnableBlockReuse)
        .def("rewind_kv_cache", &BaseKVCacheManager::rewindKVCache, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("cross_kv", &BaseKVCacheManager::isCrossKv)
        .def("store_context_blocks", &BaseKVCacheManager::storeContextBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def("store_blocks_for_reuse", &BaseKVCacheManager::storeBlocksForReuse,
            nb::call_guard<nb::gil_scoped_release>())
        .def("analyze_prefix_reuse", &BaseKVCacheManager::analyzePrefixReuse, nb::arg("unique_tokens"),
            nb::arg("llm_request"), nb::call_guard<nb::gil_scoped_release>())
        .def("get_cache_block_ids", &BaseKVCacheManager::getCacheBlockIds, nb::call_guard<nb::gil_scoped_release>())
        .def("get_batch_cache_block_ids", &BaseKVCacheManager::getBatchCacheBlockIds,
            nb::call_guard<nb::gil_scoped_release>())
        .def("flush_iteration_events", &BaseKVCacheManager::flushIterationEvents,
            nb::call_guard<nb::gil_scoped_release>())
        .def("sync_transfer_manager_with_buffer_manager", &BaseKVCacheManager::syncTransferManagerWithBufferManager,
            nb::call_guard<nb::gil_scoped_release>())
        .def("refresh_blocks", &BaseKVCacheManager::refreshBlocks, nb::call_guard<nb::gil_scoped_release>())
        .def("get_last_block_id", &BaseKVCacheManager::getLastBlockId, nb::call_guard<nb::gil_scoped_release>())
        .def("unpin_blocks_by_id", &BaseKVCacheManager::unpinBlocksById, nb::call_guard<nb::gil_scoped_release>())
        .def("reset_reuse_state", &BaseKVCacheManager::resetReuseState, nb::call_guard<nb::gil_scoped_release>())
        .def("get_priority_by_block_id", &BaseKVCacheManager::getPriorityByBlockId, nb::arg("block_id"),
            nb::arg("window_size"), nb::call_guard<nb::gil_scoped_release>());

    nb::bind_vector<CacheBlockIds>(m, "CacheBlockIds")
        .def("__getstate__", [](CacheBlockIds const& v) { return nb::make_tuple(v); })
        .def("__setstate__",
            [](CacheBlockIds& self, nb::tuple const& t)
            {
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                new (&self) CacheBlockIds(nb::cast<std::vector<std::vector<SizeType32>>>(t[0]));
            });

    nb::enum_<tbk::CacheType>(m, "CacheType")
        .value("SELF", tbk::CacheType::kSELF)
        .value("CROSS", tbk::CacheType::kCROSS)
        .value("SELFKONLY", tbk::CacheType::kSELFKONLY);

    nb::class_<tbk::KVCacheManager, tbk::BaseKVCacheManager>(m, "KVCacheManager")
        .def(nb::init<std::vector<SizeType32> const&, SizeType32, SizeType32,
                 std::map<SizeType32, std::tuple<SizeType32, SizeType32>> const&, SizeType32, SizeType32,
                 std::vector<SizeType32> const&, nvinfer1::DataType, SizeType32, int64_t, SizeType32, SizeType32, bool,
                 tbk::CacheType, std::optional<tensorrt_llm::executor::RetentionPriority>,
                 std::shared_ptr<tbk::KVCacheEventManager>, bool, bool, std::shared_ptr<tbc::KvCacheConnectorManager>,
                 bool, SizeType32, SizeType32, bool, std::optional<tbk::LinearAttentionMetadata>>(),
            nb::arg("num_kv_heads_per_layer"), nb::arg("size_per_head"), nb::arg("tokens_per_block"),
            nb::arg("blocks_per_window"), nb::arg("max_num_sequences"), nb::arg("max_beam_width"),
            nb::arg("max_attention_window_vec"), nb::arg("dtype"), nb::arg("sink_token_length"), nb::arg("stream"),
            nb::arg("max_sequence_length"), nb::arg("chunk_size"), nb::arg("enable_block_reuse") = false,
            nb::arg("cache_type") = tbk::CacheType::kSELF, nb::arg("secondary_offload_min_priority") = std::nullopt,
            nb::arg("event_manager") = nullptr, nb::arg("enable_partial_reuse") = true,
            nb::arg("copy_on_partial_reuse") = true, nb::arg("kv_connector_manager") = nullptr,
            nb::arg("enable_indexer_k_cache") = false, nb::arg("indexer_k_cache_quant_block_size") = 128,
            nb::arg("indexer_k_cache_index_head_dim") = 0, nb::arg("indexer_k_cache_use_fp4") = false,
            nb::arg("linear_attention_metadata").none() = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "scheduling_has_free_blocks",
            [](tbk::KVCacheManager& self, SizeType32 numRequired, SizeType32 windowSize)
            { return self.getBlockManager().schedulingHasFreeBlocks(numRequired, windowSize); },
            nb::arg("num_required"), nb::arg("window_size"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro(
            "is_variable_window", [](tbk::KVCacheManager& self) { return self.getBlockManager().isVariableWindow(); })
        .def("copy_linear_attention_block", &tbk::KVCacheManager::copyLinearAttentionBlock, nb::arg("llm_request"),
            nb::call_guard<nb::gil_scoped_release>());
}

void tb::BasePeftCacheManagerBindings::initBindings(nb::module_& m)
{
    nb::class_<tb::BasePeftCacheManager, PyBasePeftCacheManager>(m, "BasePeftCacheManager")
        .def("add_request_peft", &tb::BasePeftCacheManager::addRequestPeft, nb::arg("request"),
            nb::arg("try_gpu_cache") = true, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "ensure_batch",
            [](tb::BasePeftCacheManager& self, tb::RequestVector const& contextRequests,
                tb::RequestVector const& generationRequests, bool resetGpuCache)
            { return self.ensureBatch(contextRequests, generationRequests, resetGpuCache); },
            nb::arg("context_requests"), nb::arg("generation_requests"), nb::arg("reset_gpu_cache") = false,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "reset_device_cache", &tb::BasePeftCacheManager::resetDeviceCache, nb::call_guard<nb::gil_scoped_release>())
        .def("mark_request_done", &tb::BasePeftCacheManager::markRequestDone, nb::arg("request"),
            nb::arg("pause") = false, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("max_device_pages", &tb::BasePeftCacheManager::getMaxDevicePages)
        .def_prop_ro("max_host_pages", &tb::BasePeftCacheManager::getMaxHostPages)
        .def("determine_num_pages", &tb::BasePeftCacheManager::determineNumPages, nb::arg("request"),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("enabled", &tb::BasePeftCacheManager::enabled);

    nb::class_<tb::PeftCacheManager, tb::BasePeftCacheManager>(m, "PeftCacheManager")
        .def(nb::init<tb::PeftCacheManagerConfig, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            nb::arg("config"), nb::arg("model_config"), nb::arg("world_config"), nb::arg("buffer_manager"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("is_task_cached", &tb::PeftCacheManager::isTaskCached, nb::arg("taskId"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("is_task_cached_device", &tb::PeftCacheManager::isTaskCachedDevice, nb::arg("taskId"),
            nb::call_guard<nb::gil_scoped_release>()) // ;
        .def("ensure_batch_map_task_id", &tb::PeftCacheManager::ensureBatchMapTaskId, nb::arg("context_requests"),
            nb::arg("generation_requests"), nb::arg("reset_gpu_cache") = false,
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tb::NoOpPeftCacheManager, tb::BasePeftCacheManager>(m, "NoOpPeftCacheManager")
        .def(nb::init<>(), nb::call_guard<nb::gil_scoped_release>());
}
""",
    'key-files/cpp/tensorrt_llm/thop/IndexerKCacheGatherOp.cpp': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerKCacheGather.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<th::Tensor, th::Tensor> indexer_k_cache_gather_op(th::Tensor const& k_cache,
    th::Tensor const& slot_mapping_fp8, th::Tensor const& slot_mapping_scale, int64_t k_token_start, int64_t num_tokens,
    int64_t head_dim)
{
    // head_dim is the number of payload bytes per token: 128 for FP8 (one byte
    // per float8 value) or 64 for FP4 (two packed E2M1 codes per byte).
    constexpr int32_t SCALE_SIZE = 4;
    TORCH_CHECK(head_dim == 128 || head_dim == 64,
        "indexer_k_cache_gather_op head_dim must be 128 (FP8) or 64 (FP4 packed), got %d", static_cast<int>(head_dim));
    auto head_dim_i32 = static_cast<int32_t>(head_dim);

    auto device = k_cache.device();

    // Early return for empty gather. Keep the FP8 dtype on the output to stay
    // compatible with the legacy caller contract; the FP4 call site reinterprets
    // the bytes before handing them to DeepGEMM.
    if (num_tokens == 0)
    {
        auto k_fp8 = th::empty({0, head_dim_i32}, th::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));
        auto k_scale = th::empty({0, 1}, th::TensorOptions().dtype(torch::kFloat32).device(device));
        return std::make_tuple(std::move(k_fp8), std::move(k_scale));
    }

    // Validate all tensors are CUDA tensors
    TORCH_CHECK(k_cache.is_cuda() && slot_mapping_fp8.is_cuda() && slot_mapping_scale.is_cuda(),
        "All tensors must be CUDA tensors");

    // Validate tensor dimensions
    TORCH_CHECK(slot_mapping_fp8.dim() == 1, "slot_mapping_fp8 must be a 1D Tensor");
    TORCH_CHECK(slot_mapping_scale.dim() == 1, "slot_mapping_scale must be a 1D Tensor");

    // Enforce k_cache is 4D tensor
    TORCH_CHECK(k_cache.dim() == 4,
        "k_cache must be a 4D Tensor [num_blocks, block_size, 1, per_token_size], got %d dimensions",
        static_cast<int>(k_cache.dim()));

    // Validate tensor dtypes
    TORCH_CHECK(slot_mapping_fp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(slot_mapping_scale.scalar_type() == torch::kInt64, "slot_mapping_scale must be int64");

    // Validate tensors are contiguous (except k_cache which may be non-contiguous)
    // k_cache can be non-contiguous - we handle this via strides
    TORCH_CHECK(slot_mapping_fp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
    TORCH_CHECK(slot_mapping_scale.is_contiguous(), "slot_mapping_scale must be contiguous");

    // Validate slot_mapping has enough elements
    TORCH_CHECK(slot_mapping_fp8.size(0) >= k_token_start + num_tokens,
        "slot_mapping_fp8 too short for k_token_start + num_tokens");
    TORCH_CHECK(slot_mapping_scale.size(0) >= k_token_start + num_tokens,
        "slot_mapping_scale too short for k_token_start + num_tokens");

    int32_t cache_dim_0 = static_cast<int32_t>(k_cache.size(0)); // num_blocks
    int32_t cache_dim_1 = static_cast<int32_t>(k_cache.size(1)); // block_size
    int32_t cache_dim_2 = static_cast<int32_t>(k_cache.size(2)); // num_kv_heads
    int32_t cache_dim_3 = static_cast<int32_t>(k_cache.size(3)); // per_token_size

    // Validation for indexer k cache pool for DeepSeek-V3.2 constraints
    TORCH_CHECK(cache_dim_2 == 1, "k_cache dimension 2 must be 1 for DeepSeek-V3.2, got %d", cache_dim_2);

    int64_t cache_stride_0 = static_cast<int64_t>(k_cache.stride(0));
    int64_t cache_stride_1 = static_cast<int64_t>(k_cache.stride(1));
    int64_t cache_stride_2 = static_cast<int64_t>(k_cache.stride(2));
    int64_t cache_stride_3 = static_cast<int64_t>(k_cache.stride(3));

    // Allocate output buffers
    auto num_tokens_i32 = static_cast<int32_t>(num_tokens);
    auto out_fp8 = th::empty({num_tokens, head_dim_i32}, th::TensorOptions().dtype(torch::kUInt8).device(device));
    auto out_scale = th::empty({num_tokens, SCALE_SIZE}, th::TensorOptions().dtype(torch::kUInt8).device(device));

    auto stream = at::cuda::getCurrentCUDAStream(k_cache.get_device());

    tk::invokeIndexerKCacheGather(k_cache.data_ptr<uint8_t>(), slot_mapping_fp8.data_ptr<int64_t>(),
        slot_mapping_scale.data_ptr<int64_t>(), out_fp8.data_ptr<uint8_t>(), out_scale.data_ptr<uint8_t>(),
        static_cast<int32_t>(k_token_start), num_tokens_i32, head_dim_i32, SCALE_SIZE, cache_dim_0, cache_dim_1,
        cache_dim_2, cache_dim_3, cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3, stream);

    // View-cast to final dtypes (no copy). The FP8 call site keeps its existing
    // float8_e4m3fn semantics; the FP4 call site reshapes the raw bytes into
    // int8-packed nibbles after this op returns.
    auto k_fp8 = out_fp8.view(torch::kFloat8_e4m3fn);
    auto k_scale = out_scale.view(torch::kFloat32).view({num_tokens, 1});

    return std::make_tuple(std::move(k_fp8), std::move(k_scale));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_k_cache_gather_op(Tensor k_cache, Tensor slot_mapping_fp8, "
        "Tensor slot_mapping_scale, int k_token_start, int num_tokens, int head_dim) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_k_cache_gather_op", &tensorrt_llm::torch_ext::indexer_k_cache_gather_op);
}
""",
    'key-files/cpp/tensorrt_llm/thop/IndexerKCacheScatterOp.cpp': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/IndexerKCacheScatter.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void indexer_k_cache_scatter_op(th::Tensor const& k_fp8, th::Tensor const& k_scale, th::Tensor& k_cache,
    th::Tensor const& slot_mapping_fp8, th::Tensor const& slot_mapping_scale, int64_t num_tokens)
{
    // k_fp8: [>=num_tokens, head_dim] in FP8 (1 byte/element) — reinterpreted as uint8
    // k_scale: [>=num_tokens, head_dim // quant_block_size] in float32 — reinterpreted as uint8 bytes
    // slot_mapping_fp8, slot_mapping_scale: [>=num_tokens] int64 — only first num_tokens used
    // k_cache: [num_blocks, block_size, 1, per_token_size] uint8

    TORCH_CHECK(k_fp8.is_cuda() && k_scale.is_cuda() && k_cache.is_cuda() && slot_mapping_fp8.is_cuda()
            && slot_mapping_scale.is_cuda(),
        "All tensors must be CUDA tensors");

    // Validate tensor dimensions
    TORCH_CHECK(k_fp8.dim() == 2, "k_fp8 must be 2D [num_tokens, head_dim]");
    TORCH_CHECK(k_scale.dim() == 2, "k_scale must be 2D [num_tokens, scale_elements]");
    TORCH_CHECK(slot_mapping_fp8.dim() == 1, "slot_mapping_fp8 must be 1D [num_tokens]");
    TORCH_CHECK(slot_mapping_scale.dim() == 1, "slot_mapping_scale must be 1D [num_tokens]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D [num_blocks, block_size, 1, per_token_size], got %d dims",
        static_cast<int>(k_cache.dim()));

    // Validate tensor dtypes — reinterpret_cast below assumes specific element sizes
    TORCH_CHECK(k_fp8.element_size() == 1, "k_fp8 must have 1-byte elements (e.g. FP8), got %d", k_fp8.element_size());
    TORCH_CHECK(k_scale.element_size() == 4, "k_scale must have 4-byte elements (e.g. float32), got %d",
        k_scale.element_size());
    TORCH_CHECK(slot_mapping_fp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(slot_mapping_scale.scalar_type() == torch::kInt64, "slot_mapping_scale must be int64");

    TORCH_CHECK(k_fp8.is_contiguous(), "k_fp8 must be contiguous");
    TORCH_CHECK(k_scale.is_contiguous(), "k_scale must be contiguous");
    TORCH_CHECK(slot_mapping_fp8.is_contiguous(), "slot_mapping_fp8 must be contiguous");
    TORCH_CHECK(slot_mapping_scale.is_contiguous(), "slot_mapping_scale must be contiguous");

    // FP8 is 1 byte per element, so head_dim in elements == head_dim in bytes.
    int32_t const head_dim = static_cast<int32_t>(k_fp8.size(1));
    // Scale size in bytes: num_scale_elements * bytes_per_element.
    int32_t const scale_size = static_cast<int32_t>(k_scale.size(1)) * static_cast<int32_t>(k_scale.element_size());

    int32_t const cache_dim_0 = static_cast<int32_t>(k_cache.size(0));
    int32_t const cache_dim_1 = static_cast<int32_t>(k_cache.size(1));
    int32_t const cache_dim_2 = static_cast<int32_t>(k_cache.size(2));
    int32_t const cache_dim_3 = static_cast<int32_t>(k_cache.size(3));

    // Validation for indexer k cache pool.
    // head_dim = 128 (FP8 path) or 64 (FP4 path; two packed E2M1 codes per byte).
    // scale_size stays 4 bytes in both modes at head_dim=128: FP8 uses one
    // float32 per token, FP4 packs four UE8M0 exponents into one int32.
    TORCH_CHECK(cache_dim_2 == 1, "k_cache dimension 2 must be 1, got %d", cache_dim_2);
    TORCH_CHECK(
        head_dim == 128 || head_dim == 64, "k_fp8 head_dim must be 128 (FP8) or 64 (FP4 packed), got %d", head_dim);
    TORCH_CHECK(scale_size == 4, "k_scale scale_size must be 4 bytes, got %d", scale_size);

    int64_t const cache_stride_0 = static_cast<int64_t>(k_cache.stride(0));
    int64_t const cache_stride_1 = static_cast<int64_t>(k_cache.stride(1));
    int64_t const cache_stride_2 = static_cast<int64_t>(k_cache.stride(2));
    int64_t const cache_stride_3 = static_cast<int64_t>(k_cache.stride(3));

    auto stream = at::cuda::getCurrentCUDAStream(k_fp8.get_device());

    // Reinterpret k_fp8 as uint8 bytes and k_scale as raw bytes via data_ptr.
    // For slot mappings, use data_ptr directly — only the first num_tokens entries are read.
    tk::invokeIndexerKCacheScatter(reinterpret_cast<uint8_t const*>(k_fp8.data_ptr()),
        reinterpret_cast<uint8_t const*>(k_scale.data_ptr()), k_cache.data_ptr<uint8_t>(),
        slot_mapping_fp8.data_ptr<int64_t>(), slot_mapping_scale.data_ptr<int64_t>(), static_cast<int32_t>(num_tokens),
        head_dim, scale_size, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3, cache_stride_0, cache_stride_1,
        cache_stride_2, cache_stride_3, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "indexer_k_cache_scatter_op(Tensor k_fp8, Tensor k_scale, Tensor(a!) k_cache, "
        "Tensor slot_mapping_fp8, Tensor slot_mapping_scale, int num_tokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("indexer_k_cache_scatter_op", &tensorrt_llm::torch_ext::indexer_k_cache_scatter_op);
}
""",
    'key-files/cpp/tensorrt_llm/thop/fusedCatFp4Op.cpp': r"""/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/fusedCatFp4.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> fused_cat_fp4(at::Tensor const& pe, at::Tensor const& nope)
{
    CHECK_TH_CUDA(pe);
    CHECK_TH_CUDA(nope);
    TORCH_CHECK(pe.device() == nope.device(), "pe and nope must be on the same CUDA device; got pe=", pe.device(),
        ", nope=", nope.device());
    // Scope subsequent CUDA calls (empty_cuda, getCurrentCUDAStream, kernel
    // launch) to pe.device() so a caller whose current device differs cannot
    // cross-dereference the raw data pointers.
    c10::cuda::CUDAGuard deviceGuard{pe.device()};

    TORCH_CHECK(pe.scalar_type() == at::ScalarType::BFloat16, "pe must be BF16, got ", pe.scalar_type());
    TORCH_CHECK(nope.scalar_type() == at::ScalarType::BFloat16, "nope must be BF16, got ", nope.scalar_type());
    TORCH_CHECK(pe.dim() >= 2, "pe must be >= 2D, got ", pe.dim(), "D");
    TORCH_CHECK(nope.dim() >= 2, "nope must be >= 2D, got ", nope.dim(), "D");

    // Innermost dimension must be contiguous for vectorized loads.
    TORCH_CHECK(pe.stride(-1) == 1, "pe must have contiguous innermost dim (stride(-1)==1), got ", pe.stride(-1));
    TORCH_CHECK(nope.stride(-1) == 1, "nope must have contiguous innermost dim (stride(-1)==1), got ", nope.stride(-1));

    // The kernel issues 8-byte vectorized loads (int2 reinterpret of 4x BF16).
    // PyTorch CUDA storage is 256-byte aligned and split() preserves BF16
    // alignment when the split boundary is a multiple of 4 elements (all
    // current call sites satisfy this), but check explicitly so an
    // out-of-tree caller can't silently misalign the load.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(pe.data_ptr()) % 8 == 0,
        "pe.data_ptr() must be 8-byte aligned for vectorized BF16 loads (ptr=",
        reinterpret_cast<void const*>(pe.data_ptr()), ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(nope.data_ptr()) % 8 == 0,
        "nope.data_ptr() must be 8-byte aligned for vectorized BF16 loads (ptr=",
        reinterpret_cast<void const*>(nope.data_ptr()), ")");

    auto const pe_dim = static_cast<int32_t>(pe.size(-1));
    auto const nope_dim = static_cast<int32_t>(nope.size(-1));
    auto const head_dim = pe_dim + nope_dim;

    TORCH_CHECK(head_dim == 128, "head_dim (pe_dim + nope_dim) must be 128, got ", head_dim);

    // M = product of all dimensions except the last (handles 2D, 3D, 4D, etc.)
    auto const pe_M = pe.numel() / pe_dim;
    auto const nope_M = nope.numel() / nope_dim;
    TORCH_CHECK(pe_M == nope_M, "pe and nope must have same number of rows. pe: ", pe_M, ", nope: ", nope_M);
    auto const M = static_cast<int32_t>(pe_M);

    // Extract row strides — stride of the second-to-last dimension.
    // For contiguous [..., D], stride(-2) == D. For non-contiguous views from
    // split(), stride(-2) may be larger (e.g. head_dim).
    auto const pe_row_stride = static_cast<int32_t>(pe.stride(-2));
    auto const nope_row_stride = static_cast<int32_t>(nope.stride(-2));

    // Allocate output tensors.
    //   packed: [M, head_dim / 2] int8 (two E2M1 codes per byte)
    //   scale:  [M, 1] int32 (four UE8M0 exponents packed little-endian)
    at::Tensor packed_out
        = at::detail::empty_cuda({M, head_dim / 2}, at::ScalarType::Char, pe.device(), /* stride */ std::nullopt);
    at::Tensor scale_out = at::detail::empty_cuda({M, 1}, at::ScalarType::Int, pe.device(), /* stride */ std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(pe.get_device());

    tensorrt_llm::kernels::invokeFusedCatFp4(reinterpret_cast<int8_t*>(packed_out.data_ptr()),
        reinterpret_cast<int32_t*>(scale_out.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(pe.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(nope.data_ptr()), M, pe_dim, nope_dim, head_dim, pe_row_stride,
        nope_row_stride, stream);

    return {packed_out, scale_out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fused_cat_fp4(Tensor pe, Tensor nope) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_cat_fp4", &tensorrt_llm::torch_ext::fused_cat_fp4);
}
""",
    'key-files/tensorrt_llm/_torch/attention_backend/sparse/dsa.py': r'''"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs, MLAParams, PositionalEmbeddingParams)
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import \
    maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_size_in_bytes, get_sm_version, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import (fp8_fp4_mqa_logits,
                                    fp8_fp4_paged_mqa_logits, fp8_mqa_logits,
                                    fp8_paged_mqa_logits,
                                    get_paged_mqa_logits_metadata)
from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

# Optional import: fast-hadamard-transform causes CI build issues (requires wheel+torch pre-installed)
try:
    from fast_hadamard_transform import hadamard_transform
    HAS_FAST_HADAMARD = True
except ImportError:
    hadamard_transform = None
    HAS_FAST_HADAMARD = False


def _compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute flat byte indices for indexer K data and scales from global token positions.

    Shared by Indexer.prepare() (CPU) and on_update_kv_lens() (GPU) to avoid
    duplicating the slot mapping arithmetic.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (used for the scale-size formula).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.
        data_bytes_per_token: Bytes of quantized data per token in the cache
            pool. FP8 stores one byte per element (= head_dim). FP4 packs two
            E2M1 codes per byte (= head_dim // 2). Defaults to ``head_dim``
            when unset, preserving the FP8 layout for callers that haven't
            threaded the FP4 dtype through.

    Returns:
        (fp8_indices, scale_indices): Flat byte offsets into the cache pool.
    """
    if data_bytes_per_token is None:
        data_bytes_per_token = head_dim
    scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token

    block_indices_in_seq = global_positions // tokens_per_block
    pos_in_blocks = global_positions % tokens_per_block

    max_blocks = block_offsets.shape[1]
    if block_indices_in_seq.is_cuda:
        # Clamp to prevent OOB from stale token-to-seq mappings during
        # CUDA graph capture/replay with MTP + DSA.
        block_indices_in_seq = block_indices_in_seq.clamp(0, max_blocks - 1)
    else:
        assert (block_indices_in_seq < max_blocks).all(), \
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = (block_ids * block_stride + scale_base_offset +
                     pos_in_blocks * scale_size)
    return fp8_indices, scale_indices


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to activation tensor for DSA sparse attention."""
    assert x.dtype == torch.bfloat16

    if not HAS_FAST_HADAMARD:
        # Fallback: skip transformation (acceptable for test/dev)
        logger.warning_once(
            "fast-hadamard-transform not available. DSA sparse attention will skip "
            "hadamard transformation. Install with: "
            "pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git",
            key="fast_hadamard_import_missing")
        return x

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)
            ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses cached values from attn_metadata._ensure_pool_view_cached()
    to avoid redundant Python/CUDA overhead across layers.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached()

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    global_indices = torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        topk_indices,
        attn_metadata._cached_tokens_per_block,
        topk_indices.shape[1],
        attn_metadata._cached_stride_factor,
        layer_idx,
    )

    return global_indices, attn_metadata._cached_pool_view


def split_prefill_chunks(
    seq_lens: torch.Tensor,
    max_chunk_size: int,
    start_idx: int = 0,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Split prefill requests into chunks based on max_chunk_size.
    Supports two-level chunking:
    1. Request-boundary chunking: group multiple small requests into one chunk
    2. Intra-request chunking: split large requests into multiple Q-block chunks

    Args:
        seq_lens: Sequence lengths for all requests
        max_chunk_size: Maximum number of tokens per chunk
        start_idx: Starting index for prefill requests

    Returns:
        List of chunk groups, where each group is a list of chunk specs.
        Each chunk spec is (req_idx, token_start_in_req, token_end_in_req, req_cum_start)

        - For multi-request chunks: group contains multiple specs (one per request)
        - For intra-request chunks: each Q-block is a separate group with single spec
    """
    chunk_groups = []
    num_reqs = len(seq_lens)

    current_req = start_idx
    # Compute cumulative token positions
    query_start_loc_cpu = torch.cat([
        torch.zeros(1, dtype=torch.int32, device='cpu'),
        seq_lens.cumsum(dim=0).to(torch.int32)
    ])

    while current_req < num_reqs:
        seq_len = seq_lens[current_req].item()
        req_cum_start = query_start_loc_cpu[current_req].item()

        if seq_len <= max_chunk_size:
            # This request fits in one chunk - try to pack with others
            current_size = seq_len
            chunk_specs = [(current_req, 0, seq_len, req_cum_start)]
            next_req = current_req + 1

            # Try to add more requests to this chunk
            while next_req < num_reqs:
                next_seq_len = seq_lens[next_req].item()
                if next_seq_len > max_chunk_size:
                    # Next request is large, stop packing
                    break
                if current_size + next_seq_len <= max_chunk_size:
                    next_cum_start = query_start_loc_cpu[next_req].item()
                    chunk_specs.append(
                        (next_req, 0, next_seq_len, next_cum_start))
                    current_size += next_seq_len
                    next_req += 1
                else:
                    break

            # Add as one multi-request chunk group
            chunk_groups.append(chunk_specs)
            current_req = next_req
        else:
            # Large request - split into Q-blocks
            # Each Q-block is a separate chunk group (processed in separate iteration)
            num_q_blocks = (seq_len + max_chunk_size - 1) // max_chunk_size
            for q_block_idx in range(num_q_blocks):
                token_start = q_block_idx * max_chunk_size
                token_end = min(token_start + max_chunk_size, seq_len)
                q_block_spec = [(current_req, token_start, token_end,
                                 req_cum_start)]
                chunk_groups.append(q_block_spec)

            current_req += 1

    return chunk_groups


def compute_cu_seqlen_kv_bounds_with_cache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
    cached_token_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention,
    accounting for cached KV tokens.

    Args:
        seq_lens: current token lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of context tokens across all sequences in current batch
        cached_token_lens: Cached KV token lengths [num_contexts], dtype=torch.int32 (optional)

    Returns:
        cu_seqlen_ks: Start index in KV for each Q token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each Q token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Total KV lengths per request
    kv_lens = seq_lens if cached_token_lens is None else cached_token_lens + seq_lens  # [num_contexts]

    # Cumulative KV offsets: where each request's KV sequence starts in global KV space
    cu_kv_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(kv_lens, dim=0).to(torch.int32)
    ])  # [num_contexts + 1]

    # Map each Q token to its request: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32),
        seq_lens)  # [num_ctx_tokens]

    # Each Q token's KV window starts at its request's KV sequence start
    cu_seqlen_ks = cu_kv_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local Q position within each request (0-based, relative to current batch context tokens)
    cu_q_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(seq_lens, dim=0).to(torch.int32)
    ])  # [num_contexts + 1]

    global_q_positions = torch.arange(num_ctx_tokens,
                                      device=device,
                                      dtype=torch.int32)
    local_q_positions = global_q_positions - torch.repeat_interleave(
        cu_q_offsets[:-1], seq_lens)  # [num_ctx_tokens]

    if cached_token_lens is not None:
        cached_per_token = torch.repeat_interleave(cached_token_lens,
                                                   seq_lens)  # [num_ctx_tokens]
        cu_seqlen_ke = cu_seqlen_ks + cached_per_token + local_q_positions + 1  # [num_ctx_tokens]
    else:
        cu_seqlen_ke = cu_seqlen_ks + local_q_positions + 1  # [num_ctx_tokens]

    return cu_seqlen_ks, cu_seqlen_ke


@dataclass
class IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk in the indexer"""
    cu_seqlen_ks: torch.Tensor  # Attention window start for each token
    cu_seqlen_ke: torch.Tensor  # Attention window end for each token
    token_start: int  # Q token start index in batch
    token_end: int  # Q token end index in batch
    k_token_start: int  # K token start index in batch
    k_token_end: int  # K token end index in batch


class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Attention metadata for DSA (Dense Sparse Attention) with indexer state."""

    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # Topk for sparse MLA
    num_sparse_topk: int
    # max number of draft tokens
    max_draft_tokens: int = 0
    # Enable indexer skip for short sequences
    enable_indexer_skip: bool = False
    # Whether skip the indexer for context requests
    skip_indexer_for_ctx_reqs: bool = False
    # Whether skip the indexer for generation requests
    skip_indexer_for_gen_reqs: bool = False
    # Whether to use the expanded buffers for MTP support
    use_expanded_buffers_for_mtp: bool = False

    def __init__(self, *args, **kwargs):
        """Initialize DSA metadata with SM count and indexer chunk size."""
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        # Cached step-invariant values for transform_local_topk_and_prepare_pool_view.
        # These are recomputed once per step in _ensure_pool_view_cached() and
        # reused across all layers to avoid redundant Python/CUDA overhead.
        # Initialized here as plain instance attributes (not class-level
        # annotations) to stay invisible to dataclass/torch.compile introspection.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_view = None
        self._cached_stride_factor = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        super().__init__(*args, **kwargs)
        if self.sparse_attention_config.indexer_max_chunk_size is not None:
            self.indexer_max_chunk_size = self.sparse_attention_config.indexer_max_chunk_size
        else:
            self.indexer_max_chunk_size = 32768  # Default to 32K tokens for the indexer

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        super().__post_init__()
        assert isinstance(self.kv_cache_manager, DSACacheManager), \
            f"DSAtrtllmAttentionMetadata requires DSACacheManager, got {type(self.kv_cache_manager)}"

        self.num_sparse_topk = self.sparse_attention_config.index_topk
        self.enable_indexer_skip = self.sparse_attention_config.skip_indexer_for_short_seqs
        capture_graph = self.is_cuda_graph

        self.indexer_k_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq],
            cache_name="indexer_k_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_indexer_k_cache_block_offsets = torch.zeros_like(
            self.indexer_k_cache_block_offsets,
            device='cpu',
            pin_memory=prefer_pinned(),
        )

        if not self.enable_context_mla_with_cached_kv:
            self.ctx_cached_token_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1, ),
                cache_name="ctx_cached_token_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_cached_token_indptr = torch.zeros_like(
                self.ctx_cached_token_indptr,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
            self.ctx_kv_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1, ),
                cache_name="ctx_kv_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_kv_indptr = torch.zeros_like(
                self.ctx_kv_indptr,
                device='cpu',
                pin_memory=prefer_pinned(),
            )

        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
        # These buffers will be allocated dynamically in Indexer.prepare() based on actual total_kv_len to save memory.
        if self.enable_context_mla_with_cached_kv:
            self.slot_mapping_fp8_fullkv = None
            self.slot_mapping_scale_fullkv = None

        # New generation buffers for dsa
        self.gen_cached_token_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1, ),
            cache_name="gen_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_cached_token_indptr = torch.zeros_like(
            self.gen_cached_token_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.gen_kv_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1, ),
            cache_name="gen_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_kv_indptr = torch.zeros_like(
            self.gen_kv_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # Indexer metadata
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="slot_mapping_fp8",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_fp8 = torch.zeros_like(
            self.slot_mapping_fp8,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.slot_mapping_scale = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="slot_mapping_scale",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_scale = torch.zeros_like(
            self.slot_mapping_scale,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # Per-token request index buffer for topk_indices conversion
        self.req_idx_per_token = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Block table for topk_indices conversion (shared for context and generation)
        self.block_table = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests, self.kv_cache_manager.max_blocks_per_seq),
            cache_name="block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.scheduler_metadata_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # When MTP runs without the expanded-tokens path, the same forward step
        # alternates between full-window calls (next_n == 1 + max_draft_tokens)
        # and per-token draft calls (next_n == 1). The 2D DeepGEMM metadata
        # API encodes next_n into the schedule, so the precomputed schedule
        # for one shape cannot be reused for the other. Maintain a second
        # buffer holding the schedule for the full next_n window; the draft
        # path keeps using `scheduler_metadata_buffer`. Always allocate (a
        # few KB) so transitions in `update_spec_dec_param` don't have to
        # special-case its existence.
        self.scheduler_metadata_buffer_full_next_n = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_full_next_n",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Pre-allocated 2D kv_lens buffer for the new DeepGEMM 2D context_lens
        # API. Shape: (max_num_sequences, 1 + max_draft_tokens). Each row
        # broadcasts the same kv_len across next_n positions; kernel reads a
        # slice per forward. Avoids per-forward .expand().contiguous()
        # allocations that would break CUDA graphs.
        self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        self.cu_seqlen_ks = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="cu_seqlen_ks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ke = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="cu_seqlen_ke",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Topk indices buffer to support skip indexer for requests with short sequence lengths
        if self.enable_indexer_skip:
            self.topk_indices_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
        # Per-layer persistent buffers for heuristic TopK pre_idx.
        # Indexed by [local_layer_idx, generation_position, :].
        # The graph captures reads/writes on these stable-address buffers;
        # each replay's write becomes the next replay's read (feedback loop).
        self.enable_heuristic_topk = (
            self.sparse_attention_config.enable_heuristic_topk
            and get_sm_version() >= 100)
        if self.enable_heuristic_topk:
            num_local_layers = self.kv_cache_manager.num_local_layers
            self.heuristic_prev_topk = self.get_empty(
                self.cuda_graph_buffers,
                (num_local_layers, self.max_num_sequences,
                 self.num_sparse_topk),
                cache_name="heuristic_prev_topk",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            # Zero-initialize so the first decode step's pre_idx (kernel
            # adds +1 offset) points to index 1 — a valid but benign candidate.
            # Without this, uninitialized memory produces random hint indices.
            self.heuristic_prev_topk.zero_()
            # Scratch buffer for heuristic TopK kernel output values.
            # Pre-allocated with stable address for CUDA Graph compatibility
            # (replaces cudaMallocAsync/cudaFreeAsync inside the kernel launcher).
            # Shape: [max_gen_tokens, topK] where max_gen_tokens = max_batch * (1 + max_draft).
            max_gen_tokens = self.max_num_sequences * (1 +
                                                       self.max_draft_tokens)
            self.heuristic_scratch_values = self.get_empty(
                self.cuda_graph_buffers,
                (max_gen_tokens, self.num_sparse_topk),
                cache_name="heuristic_scratch_values",
                dtype=torch.float32,
                capture_graph=capture_graph,
            )

        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    def _create_kv_lens_2d_buffer(self, capture_graph=False):
        """Pre-allocated buffer for the DeepGEMM 2D context_lens API.

        Avoids per-forward .expand().contiguous() allocations that break CUDA
        graphs. The buffer is written in-place via .copy_() inside
        on_update_kv_lens so its address stays stable across replays.
        """
        self.kv_lens_cuda_2d = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, 1 + self.max_draft_tokens),
            cache_name="kv_lens_cuda_2d",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
        """Create expanded KV-length and block-table buffers for speculative decoding."""
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self.max_draft_tokens), ),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self.max_draft_tokens),
                self.kv_cache_manager.max_blocks_per_seq
            ],
            cache_name="block_table_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_block_table_expanded = torch.zeros_like(
            self.block_table_expanded,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # This function is only used to create the expanded buffers when the max_draft_tokens is changed.
    # TODO: remove this function once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def update_spec_dec_param(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        model_is_wrapped: bool = False,
        spec_metadata: Optional['SpecMetadata'] = None,
        spec_tree_manager: Optional['SpecTreeManager'] = None,
    ):
        """Update speculative decoding parameters and create expanded buffers."""
        super().update_spec_dec_param(batch_size, is_spec_decoding_enabled,
                                      is_spec_dec_tree,
                                      is_spec_dec_dynamic_tree, max_draft_len,
                                      max_total_draft_tokens, model_is_wrapped,
                                      spec_metadata, spec_tree_manager)
        self.max_draft_tokens = max_draft_len
        capture_graph = self.is_cuda_graph
        if self.kv_lens_cuda_2d.shape[1] != 1 + self.max_draft_tokens:
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        init_shape = self.kv_lens_expanded_host.shape[0]
        if self.max_num_sequences * (1 + self.max_draft_tokens) != init_shape:
            self.create_expanded_buffers(capture_graph=capture_graph)
            # Resize heuristic scratch buffer for new max_draft_tokens.
            if self.enable_heuristic_topk:
                max_gen_tokens = self.max_num_sequences * (
                    1 + self.max_draft_tokens)
                self.heuristic_scratch_values = self.get_empty(
                    self.cuda_graph_buffers,
                    (max_gen_tokens, self.num_sparse_topk),
                    cache_name="heuristic_scratch_values",
                    dtype=torch.float32,
                    capture_graph=capture_graph,
                )

    def _invalidate_pool_view_cache(self):
        """Invalidate the cached pool view and related step-invariant values.

        Must be called at the start of each forward step (in prepare()) so that
        _ensure_pool_view_cached() recomputes them for the new batch.
        """
        self._pool_cache_valid = False

    def _ensure_pool_view_cached(self):
        """Compute and cache values used by
        transform_local_topk_and_prepare_pool_view().

        These values (pool view, stride factor, block table slices, request
        index slices) are constant across all layers sharing the same KV pool
        and batch dimensions within a forward pass. Caching them avoids
        redundant Python/CUDA overhead per layer.

        Safety: _invalidate_pool_view_cache() is called unconditionally at the
        start of every step (prepare() and on_update_kv_lens()), so the boolean
        flag is always cleared before the first per-layer call within a step.
        """
        if self._pool_cache_valid and self._cached_kv_mgr_id == id(
                self.kv_cache_manager):
            return

        pool = self.kv_cache_manager.get_unique_primary_pool()
        kv_cache_manager = self.kv_cache_manager
        num_blocks, num_layers, _, _ = pool.shape
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_pool_view = pool.squeeze(2).view(-1, 1, head_dim)
        self._cached_stride_factor = (num_layers *
                                      self._cached_tokens_per_block)
        self._cached_block_table_ctx = self.block_table[:self.num_contexts]
        self._cached_block_table_gen = self.block_table[self.num_contexts:self.
                                                        num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[:self.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[self.num_ctx_tokens:self.num_tokens] -
            self.num_contexts)
        self._cached_kv_mgr_id = id(kv_cache_manager)
        self._pool_cache_valid = True

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        # get position ids
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts  # Shape: [batch_size]
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends,
                                           global_indices,
                                           side='right')
        repeated_offsets = per_seq_offsets[batch_indices]
        position_ids = global_indices + repeated_offsets
        # get the dense topk indices with causal mask
        range_row = torch.arange(self.num_sparse_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self,
                                   kv_lens,
                                   device=False):  # device=False means use CPU
        """Prepare dense TopK indices for short sequences that skip the indexer."""

        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    ctx_range, :] = self._get_dense_topk_indices(
                        self.seq_lens[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens)
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :],
                    non_blocking=True)

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    gen_range, :] = self._get_dense_topk_indices(
                        self.seq_lens[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens)
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :],
                    non_blocking=True)

    def _get_pool_block_indices(self) -> torch.Tensor:
        """Extract memory pool block indices from host_kv_cache_block_offsets.

        The C++ setOffsets() encodes offsets as:
            encoded = memPoolBlockIndex * numLayers * kvFactor
        For SELFKONLY (MLA/DSA), kvFactor=1, so:
            memPoolBlockIndex = encoded // num_local_layers

        Returns a (num_seqs, max_blocks_per_seq) int32 CPU tensor with valid
        pool indices clamped to [0, blocks_in_primary_pool - 1].
        """
        num_local_layers = self.kv_cache_manager.num_local_layers
        max_pool_idx = self.kv_cache_manager.blocks_in_primary_pool - 1
        # DSA uses SELFKONLY mode where only key cache is stored (kv_factor=1).
        # host_kv_cache_block_offsets shape: (num_pools, max_batch*beam, 2, max_blocks_per_seq)
        # Note: dim=2 is always 2 in the tensor layout (K and V slots), but for
        # SELFKONLY only the K slot (index 0) contains valid data.
        assert self.kv_cache_manager.kv_factor == 1, \
            f"DSA requires SELFKONLY mode (kv_factor=1), got kv_factor={self.kv_cache_manager.kv_factor}"
        # Pool 0, first num_seqs entries, field 0 (key offsets)
        encoded = self.kv_cache_manager.host_kv_cache_block_offsets[
            0, :self.num_seqs, 0, :]
        pool_indices = encoded // num_local_layers
        # Clamp for safety: handles garbage padding from torch.empty in uninitialized slots
        pool_indices = pool_indices.clamp(min=0,
                                          max=max_pool_idx).to(torch.int32)
        return pool_indices

    def prepare(self):
        """Prepare DSA metadata: compute slot mappings, block tables, and prefill chunks."""
        super().prepare()
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        )
        if self.enable_helix:
            # For Helix CP, inactive ranks only attend to previously cached
            # tokens (no new token appended), while active ranks add new tokens.
            # This mirrors the kv_lens logic in TrtllmAttentionMetadata.prepare().
            active_rank = ~self.helix_is_inactive_rank_cpu[:self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        # Prepare to support skip indexer
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = kv_lens[:self.num_contexts].max(
            ).item() <= self.num_sparse_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_ctx_reqs = False

        if self.num_generations > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = kv_lens[
                self.num_contexts:self.num_seqs].max().item(
                ) <= self.num_sparse_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

        # Build indexer_k_cache_block_offsets using pool block indices derived
        # from host_kv_cache_block_offsets (populated by super().prepare()).
        # This correctly resolves block IDs to memory pool indices, which is
        # required when host cache offload is enabled (block IDs != pool indices
        # for onboarded secondary blocks).
        if self.kv_cache_manager is not None:
            pool_indices = self._get_pool_block_indices()
            self.host_indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                pool_indices)
            self.indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                self.host_indexer_k_cache_block_offsets[:self.num_seqs],
                non_blocking=True)
            # Safety clamp: prevent OOB from CUDA graph padding entries which
            # may contain stale negative or out-of-range values after block
            # eviction/onboarding with host cache offload.
            self.indexer_k_cache_block_offsets.clamp_(min=0)

        # Build req_idx_per_token for topk_indices conversion
        host_req_idx_per_token = torch.repeat_interleave(torch.arange(
            self.num_seqs, dtype=torch.int32),
                                                         self.seq_lens,
                                                         dim=0)
        self.req_idx_per_token[:self.num_tokens].copy_(host_req_idx_per_token,
                                                       non_blocking=True)

        # Build block_table for topk_indices conversion (actual block allocation)
        if self.kv_cache_manager is not None:
            tokens_per_block = self.kv_cache_manager.tokens_per_block
            num_blocks_per_seq = (kv_lens[:self.num_seqs] + tokens_per_block -
                                  1) // tokens_per_block
            max_blocks_used = num_blocks_per_seq.max().item(
            ) if self.num_seqs > 0 else 1
            # pool_indices already has correct values; set padding to -1
            host_block_table = pool_indices[:, :max_blocks_used].clone()
            for i in range(self.num_seqs):
                if num_blocks_per_seq[i] < max_blocks_used:
                    host_block_table[i, num_blocks_per_seq[i]:] = -1
            # Copy to GPU
            self.block_table[:self.num_seqs, :max_blocks_used].copy_(
                host_block_table, non_blocking=True)

        # For mla_rope_append_paged_kv_assign_q
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:self.
                                                           num_contexts].sum(
                                                           ).item()
            self.max_ctx_kv_len = kv_lens[:self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:self.num_contexts].max().item(
            )
            # context cached token indptr
            torch.cumsum(
                cached_token_lens[:self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1:self.num_contexts + 1])
            self.ctx_cached_token_indptr[:self.num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[:self.num_contexts + 1],
                non_blocking=True)
            # context kv indptr
            torch.cumsum(kv_lens[:self.num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_ctx_kv_indptr[1:self.num_contexts + 1])
            self.ctx_kv_indptr[:self.num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[:self.num_contexts + 1],
                non_blocking=True)
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

        if self.num_generations > 0:
            self.max_gen_seq_len = self.seq_lens[self.num_contexts:self.
                                                 num_seqs].max().item()
            # generation cached token indptr
            torch.cumsum(
                cached_token_lens[self.num_contexts:self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_cached_token_indptr[1:self.num_generations +
                                                      1])
            self.gen_cached_token_indptr[:self.num_generations + 1].copy_(
                self.host_gen_cached_token_indptr[:self.num_generations + 1],
                non_blocking=True)
            # generation kv indptr
            torch.cumsum(kv_lens[self.num_contexts:self.num_seqs],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_gen_kv_indptr[1:self.num_generations +
                                                     1])
            self.gen_kv_indptr[:self.num_generations + 1].copy_(
                self.host_gen_kv_indptr[:self.num_generations + 1],
                non_blocking=True)
        else:
            self.max_gen_seq_len = 0

        # Because the fp8_paged_mqa_logits only supports seq_len == 1/2/4 (i.e., max_draft_tokens == 0/1/3) on sm100, and
        # seq_len == 1/2 (i.e., max_draft_tokens == 0/1) on sm90, for other cases, we need to flatten the q tensor and
        # expand the kv_lens and block_table for MTP support.
        # TODO:
        # - No distinction between sm90 and sm100 is needed once MTP3 is supported on sm90.
        # - Remove this once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
        self.use_expanded_buffers_for_mtp = (
            (self.max_draft_tokens > 1 and get_sm_version() == 90)
            or ((self.max_draft_tokens == 2 or self.max_draft_tokens > 3)
                and get_sm_version() >= 100))
        if self.use_expanded_buffers_for_mtp:
            # Expand kv_lens_cuda (only generation)
            num_tokens = self.num_generations * (1 + self.max_draft_tokens)
            gen_kv_lens = kv_lens[self.num_contexts:self.num_seqs]
            gen_kv_lens_expanded = torch.stack([gen_kv_lens] *
                                               (1 + self.max_draft_tokens),
                                               dim=0)
            gen_kv_lens_expanded = gen_kv_lens_expanded.transpose(
                0, 1).contiguous().flatten()
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self.kv_lens_expanded_cuda[:num_tokens].copy_(
                self.kv_lens_expanded_host[:num_tokens], non_blocking=True)

            # Expand indexer_k_cache_block_offsets (only generation)
            # host_indexer_k_cache_block_offsets already contains correct pool
            # indices from _get_pool_block_indices() above.
            if self.kv_cache_manager is not None and self.num_generations > 0:
                max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                gen_block_tensor = self.host_indexer_k_cache_block_offsets[
                    self.num_contexts:self.num_seqs, :max_len]
                expanded_blocks = gen_block_tensor.repeat_interleave(
                    1 + self.max_draft_tokens, dim=0)
                self.host_block_table_expanded[:num_tokens, :max_len].copy_(
                    expanded_blocks, non_blocking=True)
                self.block_table_expanded[:num_tokens].copy_(
                    self.host_block_table_expanded[:num_tokens],
                    non_blocking=True)
                self.block_table_expanded.clamp_(min=0)

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

    def on_update_kv_lens(self):
        """Refresh indexer slot mappings after KV lengths change at runtime."""
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside _preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        # _preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        if self.kv_cache_manager is not None and self.num_tokens > 0:
            seq_lens = self.seq_lens_cuda[:self.num_seqs]
            # Runtime cached lengths after overlap/spec-dec correction.
            start_positions = self.kv_lens_cuda[:self.num_seqs] - seq_lens

            # Reuse request-per-token mapping prepared in metadata.prepare().
            # This avoids repeat_interleave in graph-capture mode.
            req_indices = self.req_idx_per_token[:self.num_tokens].to(
                dtype=torch.int64)
            seq_starts = torch.cumsum(
                seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
            token_offsets = torch.arange(
                self.num_tokens, device=seq_lens.device,
                dtype=torch.int64) - seq_starts[req_indices]

            global_positions = start_positions[req_indices] + token_offsets
            # Under FP4 the indexer cache stores two E2M1 codes per byte, so
            # the per-token data footprint is head_dim // 2; otherwise it is
            # head_dim (one FP8 byte per element). Feed the real byte count
            # into _compute_slot_mappings so scatter/gather see offsets that
            # match the pool layout produced by createIndexerKCachePools.
            use_fp4 = self.kv_cache_manager.use_fp4
            index_head_dim = self.kv_cache_manager.index_head_dim
            data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                index_head_dim,
                self.kv_cache_manager.tokens_per_block,
                self.kv_cache_manager.quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )
            self.slot_mapping_fp8[:self.num_tokens] = fp8_indices
            self.slot_mapping_scale[:self.num_tokens] = scale_indices

        if self.num_generations > 0:
            torch.cumsum(
                self.kv_lens_cuda[self.num_contexts:self.
                                  num_seqs],  # num_contexts should be 0
                dim=0,
                dtype=torch.int64,
                out=self.gen_kv_indptr[1:self.num_generations + 1])
            torch.cumsum(
                (self.kv_lens_cuda[self.num_contexts:self.num_seqs] -
                 self.seq_lens_cuda[self.num_contexts:self.num_seqs]),
                dim=0,
                dtype=torch.int64,
                out=self.gen_cached_token_indptr[1:self.num_generations + 1])
            # Write 2D kv_lens in-place (broadcast same kv_len across next_n
            # positions). .expand() returns a view and .copy_() writes into the
            # pre-allocated destination, so this is CUDA-graph-friendly.
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts:self.num_seqs]
            next_n_cap = self.kv_lens_cuda_2d.shape[1]
            self.kv_lens_cuda_2d[:self.num_generations, :next_n_cap].copy_(
                gen_kv_lens.unsqueeze(-1).expand(-1, next_n_cap))
            # Build the next_n=1 schedule (used by MTP draft layers and any
            # non-MTP forward). Reshape the contiguous gen slice of
            # kv_lens_cuda to (num_gen, 1) — slicing kv_lens_cuda_2d's first
            # column would be non-contiguous and would fail the metadata
            # kernel's is_contiguous assertion.
            context_lens_next_n1 = gen_kv_lens.view(-1, 1)
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                context_lens_next_n1, self.kv_cache_manager.tokens_per_block,
                self.num_sms)
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer,
                                                 non_blocking=True)
            # When MTP is on without the expanded-tokens path, also populate
            # the full-next_n schedule for the main forward call. The metadata
            # kernel reads next_n from context_lens.size(1), so we must pass
            # the wider slice here.
            if (self.max_draft_tokens > 0
                    and not self.use_expanded_buffers_for_mtp):
                context_lens_full_next_n = self.kv_lens_cuda_2d[:self.
                                                                num_generations, :
                                                                next_n_cap]
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    context_lens_full_next_n,
                    self.kv_cache_manager.tokens_per_block, self.num_sms)
                self.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True)
            if self.use_expanded_buffers_for_mtp:
                num_draft_tokens = 1 + self.max_draft_tokens
                num_tokens = self.num_generations * num_draft_tokens
                kv_lens_expanded = torch.stack([gen_kv_lens] * num_draft_tokens,
                                               dim=0)
                self.kv_lens_expanded_cuda[:num_tokens] = \
                    kv_lens_expanded.transpose(0, 1).contiguous().flatten()
                # New API requires 2D; each expanded token becomes a (1,) row.
                kv_lens_expanded_2d = self.kv_lens_expanded_cuda[:
                                                                 num_tokens].view(
                                                                     -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, self.kv_cache_manager.tokens_per_block,
                    self.num_sms)
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def update_for_spec_dec(self):
        """Reset context/generation counters and refresh slot mappings for speculative decoding."""
        super().update_for_spec_dec()
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1

        # device
        self.on_update_kv_lens()


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor,
           s: float) -> torch.Tensor:
    """Scale attention weights by quantization scale and constant factor."""
    return weights * q_scale.squeeze(-1) * s


@maybe_compile(dynamic=True)
def _to_float(hidden_states: torch.Tensor) -> torch.Tensor:
    """Cast hidden states to float32 for TF32 GEMM computation."""
    return hidden_states.float()


@contextmanager
def _tf32_matmul_enabled():
    """Temporarily enable TF32 tensor cores for FP32 matmul in this scope.

    Forces PyTorch/cuBLASLt to use CUBLAS_COMPUTE_32F_FAST_TF32, which
    guarantees TF32 tensor cores. Plain CUBLAS_COMPUTE_32F (used by
    torch.ops.trtllm.cublas_mm) falls back to SIMT SGEMM on CUDA cores
    based on cuBLASLt heuristics for small M.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class Indexer(nn.Module):
    """DSA sparse attention indexer that selects top-K KV cache entries per token."""

    def __init__(self,
                 quant_config: Optional[QuantConfig],
                 pos_embd_params: Optional[PositionalEmbeddingParams],
                 mla_params: Optional[MLAParams],
                 skip_create_weights_in_init: bool,
                 sparse_attention_config: "SparseAttentionConfig",
                 dtype: Optional[torch.dtype],
                 layer_idx: int = 0,
                 aux_stream: Optional[torch.cuda.Stream] = None):
        """Initialize indexer with projection weights, norms, and TopK configuration."""
        super().__init__()
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_attention_config.index_n_heads  # 64
        self.head_dim = sparse_attention_config.index_head_dim  # 128
        self.index_topk = sparse_attention_config.index_topk  # 2048
        self.layer_idx = layer_idx

        self.wq_b = Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)
        self.wk = Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)

        # Fused wk + weights_proj weight for single F.linear FP32 GEMM under allow_tf32.
        # Maps to TF32 tensor cores on Ampere+.
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

        indexer_rope_interleave = getattr(sparse_attention_config,
                                          'indexer_rope_interleave', False)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=not indexer_rope_interleave,
        )

        self.softmax_scale = self.head_dim**-0.5
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        # indexer_k_dtype controls both Q and K precision. DeepGEMM's
        # fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels only dispatch
        # to FP4xFP4 or FP8xFP8 (no mixed-precision variant). The DeepGEMM
        # kernel asserts SM100 + head_dim=128 at launch time under FP4.
        self.use_fp4 = sparse_attention_config.indexer_k_dtype == "fp4"
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.use_cute_dsl_topk = (sparse_attention_config.use_cute_dsl_topk
                                  and IS_CUTLASS_DSL_AVAILABLE)
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = (
            sparse_attention_config.enable_heuristic_topk
            and get_sm_version() >= 100)

        if self.use_cute_dsl_topk and layer_idx == 0:
            from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

            # the dtype of topk input tensor, which is float32 now.
            # Note, need to update it if the dtype of topk input tensor is changed.
            cute_dsl_custom_ops.warmup_cute_dsl_indexer_topk(
                dtype=torch.float32, top_k=self.index_topk)

        if self._enable_heuristic_topk and layer_idx == 0:
            # Populate static caches (sm_count, L2 cache size) inside the C++
            # Scheme X dispatcher before any CUDA Graph capture so the host
            # attribute queries do not end up frozen into a captured graph.
            from tensorrt_llm._torch.custom_ops import cpp_custom_ops
            cpp_custom_ops.warmup_heuristic_topk_decode(top_k=self.index_topk)

    def post_load_weights(self):
        """Fuse wk + weights_proj into single FP32 weight for F.linear GEMM under allow_tf32 (TF32 tensor cores on Ampere+)."""
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0)

    @staticmethod
    def prepare_one_prefill_chunk(
        metadata: DSAtrtllmAttentionMetadata,
        chunk_specs: List[Tuple[int, int, int, int]],
    ) -> IndexerPrefillChunkMetadata:
        """
        Build metadata for one prefill chunk for indexer forward pass.
        Handles both multi-request chunks and intra-request Q-block chunks.

        Args:
            metadata: Attention metadata
            chunk_specs: List of (req_idx, token_start_in_req, token_end_in_req, req_cum_start)
                        - token_start_in_req, token_end_in_req are indices into current batch context tokens
                        - For multi-request: multiple specs from different requests (full requests)
                        - For intra-request: single spec from one request's Q-block

        Note: Cached token counts are derived from metadata.host_ctx_cached_token_indptr
        """
        device = metadata.cu_seqlen_ks.device
        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[
                0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # Get cached token count for this request from metadata
            num_cached = (
                metadata.host_ctx_cached_token_indptr[req_idx + 1] -
                metadata.host_ctx_cached_token_indptr[req_idx]).item()

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request's current tokens
            # K tokens [0:num_cached + token_end_in_req] within the request (causal attention)
            cu_seqlen_ks = torch.zeros(num_q_tokens,
                                       dtype=torch.int32,
                                       device='cpu')
            cu_seqlen_ke = torch.arange(token_start_in_req + 1,
                                        token_end_in_req + 1,
                                        dtype=torch.int32,
                                        device='cpu') + num_cached

            # Q token range in batch (indices into context tokens in the current batch)
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range: index into full KV slot mapping (cached + current batch context tokens)
            kv_offset_in_extended = metadata.host_ctx_kv_indptr[req_idx].item()
            total_kv_for_req = num_cached + token_end_in_req
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_for_req

        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            req_cached_lens = []
            first_req_idx = chunk_specs[0][0]

            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                req_seq_lens.append(token_end_in_req - token_start_in_req)
                # Get cached token count from metadata
                num_cached = (
                    metadata.host_ctx_cached_token_indptr[req_idx + 1] -
                    metadata.host_ctx_cached_token_indptr[req_idx]).item()
                req_cached_lens.append(num_cached)

            req_seq_lens_tensor = torch.tensor(req_seq_lens,
                                               dtype=torch.int32,
                                               device='cpu')
            req_cached_lens_tensor = torch.tensor(req_cached_lens,
                                                  dtype=torch.int32,
                                                  device='cpu')
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                req_seq_lens_tensor, len(chunk_specs), num_q_tokens,
                req_cached_lens_tensor)

            # Global Q token ranges (indices into ctx tokens in the current batch)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = token_start + num_q_tokens

            # K token range: index into full kv slot mapping (cached + current ctx tokens within the batch)
            kv_offset_in_extended = metadata.host_ctx_kv_indptr[
                first_req_idx].item()
            total_kv_len = sum(req_seq_lens_tensor +
                               req_cached_lens_tensor).item()
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_len

        assert cu_seqlen_ks.shape[0] == num_q_tokens == token_end - token_start, \
            f"Indexer.prepare_one_prefill_chunk - cu_seqlen_ks length mismatch: {cu_seqlen_ks.shape[0]} != {num_q_tokens}"
        assert cu_seqlen_ke.shape[0] == num_q_tokens == token_end - token_start, \
            f"Indexer.prepare_one_prefill_chunk - cu_seqlen_ke length mismatch: {cu_seqlen_ke.shape[0]} != {num_q_tokens}"

        return IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks.to(device, non_blocking=True),
            cu_seqlen_ke=cu_seqlen_ke.to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
        )

    @staticmethod
    def recompute_slot_mappings(metadata: DSAtrtllmAttentionMetadata):
        """Recompute only slot_mapping_fp8/scale from the current block offsets.

        This is the subset of prepare() that maps each token to its flat cache
        position.  It is safe to call in isolation (e.g. during draft KV-cache
        replay) because it only touches slot-mapping buffers and reads
        block-offset / sequence metadata that the caller has already set up.
        """
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None or not hasattr(kv_cache_manager,
                                                   'index_head_dim'):
            return

        seq_lens = metadata.seq_lens
        head_dim = kv_cache_manager.index_head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        quant_block_size = kv_cache_manager.quant_block_size
        use_fp4 = kv_cache_manager.use_fp4
        # FP4 packs two E2M1 codes per byte; FP8 stores one byte per element.
        data_bytes_per_token = head_dim // 2 if use_fp4 else head_dim
        cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        total_tokens = seq_lens.sum().item()

        start_positions = torch.tensor(cached_tokens, dtype=torch.int32)
        batch_size = len(metadata.request_ids)

        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device='cpu'), seq_lens)

        token_offsets = torch.cat([
            torch.arange(seq_lens[i].item(), dtype=torch.int64, device='cpu')
            for i in range(batch_size)
        ])

        global_positions = start_positions[req_indices] + token_offsets

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        metadata.host_slot_mapping_fp8[:total_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_tokens], non_blocking=True)
        metadata.slot_mapping_scale[:total_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_tokens], non_blocking=True)

    @staticmethod
    def prepare(metadata: DSAtrtllmAttentionMetadata):
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """
        kv_cache_manager = metadata.kv_cache_manager
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        tokens_per_block = kv_cache_manager.tokens_per_block

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
            host_cached_tokens = torch.tensor(cached_tokens[:num_contexts],
                                              dtype=torch.int32,
                                              device='cpu')

            # When MLA chunked prefill is active, it already handles chunking
            # Indexer should just process the current MLA chunk as a single chunk
            has_mla_chunked_prefill = (
                metadata.enable_context_mla_with_cached_kv
                and metadata.runtime_features.chunked_prefill)

            if has_mla_chunked_prefill:
                # MLA chunked prefill is active - use single-chunk pattern for
                # indexer prefill chunks.
                chunk_specs = [(i, 0, host_seq_lens[i].item(),
                                host_seq_lens[:i].sum().item() if i > 0 else 0)
                               for i in range(num_contexts)]
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                ]
            else:
                # Use indexer's own chunking logic to prevent L^2 complexity of indexer MQA logits computation for long sequences.
                # This is only used when MLA chunked prefill is not enabled.
                chunk_groups = split_prefill_chunks(
                    host_seq_lens,
                    metadata.indexer_max_chunk_size,
                    start_idx=0,
                )

                if len(chunk_groups
                       ) > 1 or metadata.enable_context_mla_with_cached_kv:
                    metadata.indexer_prefill_chunks = [
                        Indexer.prepare_one_prefill_chunk(
                            metadata,
                            chunk_specs,
                        ) for chunk_specs in chunk_groups
                    ]
                else:
                    metadata.indexer_prefill_chunks = None

            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens, num_contexts, num_ctx_tokens, host_cached_tokens)

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(host_cu_seqlen_ks,
                                                         non_blocking=True)
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(host_cu_seqlen_ke,
                                                         non_blocking=True)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            if not metadata.use_expanded_buffers_for_mtp:
                # Write 2D kv_lens (broadcast same kv_len across next_n positions).
                gen_seq_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                next_n_cap = metadata.kv_lens_cuda_2d.shape[1]
                metadata.kv_lens_cuda_2d[:num_generations, :next_n_cap].copy_(
                    gen_seq_lens.unsqueeze(-1).expand(-1, next_n_cap))
                # Build the next_n=1 schedule (used by MTP draft layers).
                # Use the contiguous 1D gen slice reshaped to (num_gen, 1);
                # slicing kv_lens_cuda_2d's first column would be a strided
                # view that fails the metadata kernel's contiguous assertion.
                context_lens_next_n1 = gen_seq_lens.view(-1, 1)
                scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                    context_lens_next_n1, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer.copy_(
                    scheduler_metadata_buffer, non_blocking=True)
                # MTP main forward uses next_n = 1 + max_draft_tokens; build
                # a separate schedule because the metadata kernel reads next_n
                # from context_lens.size(1).
                if metadata.max_draft_tokens > 0:
                    context_lens_full_next_n = metadata.kv_lens_cuda_2d[:
                                                                        num_generations, :
                                                                        next_n_cap]
                    scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                        context_lens_full_next_n, tokens_per_block,
                        metadata.num_sms)
                    metadata.scheduler_metadata_buffer_full_next_n.copy_(
                        scheduler_metadata_buffer_full_next_n,
                        non_blocking=True)
            else:
                # Expand schedule metadata buffer (only generation). The new
                # DeepGEMM API requires 2D; each expanded token becomes a (1,)
                # row.
                num_tokens = metadata.num_generations * (
                    1 + metadata.max_draft_tokens)
                kv_lens_expanded_2d = metadata.kv_lens_expanded_cuda[:
                                                                     num_tokens].view(
                                                                         -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)

        # Compute slot_mapping for all requests (both context and generation)
        Indexer.recompute_slot_mappings(metadata)

        # When chunked prefill or KVCache reuse is enabled, we need to gather the full KV for indexer's logit computation.
        # Indexer's own chunking does not need full KV gathering, instead it gathers only the current chunk with loop-based gathering.
        _need_full_kv_gathering = num_contexts > 0 and metadata.enable_context_mla_with_cached_kv
        if _need_full_kv_gathering:
            head_dim = kv_cache_manager.index_head_dim
            quant_block_size = kv_cache_manager.quant_block_size
            use_fp4 = kv_cache_manager.use_fp4
            data_bytes_per_token = head_dim // 2 if use_fp4 else head_dim
            cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
            start_positions = torch.tensor(cached_tokens, dtype=torch.int32)

            total_kv_len = metadata.host_ctx_kv_indptr[num_contexts].item()
            total_kv_per_request = seq_lens[:
                                            num_contexts] + start_positions[:
                                                                            num_contexts]
            host_slot_mapping_fp8_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned())
            host_slot_mapping_scale_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned())

            fullkv_req_indices = torch.repeat_interleave(
                torch.arange(num_contexts, dtype=torch.int64, device='cpu'),
                total_kv_per_request)

            kv_positions = torch.cat([
                torch.arange(total_kv_per_request[i].item(),
                             dtype=torch.int64,
                             device='cpu') for i in range(num_contexts)
            ])

            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                kv_positions,
                metadata.host_indexer_k_cache_block_offsets,
                fullkv_req_indices,
                head_dim,
                tokens_per_block,
                quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )

            host_slot_mapping_fp8_fullkv[:total_kv_len] = fp8_flat_indices
            host_slot_mapping_scale_fullkv[:total_kv_len] = scale_flat_indices

            assert len(fp8_flat_indices) == total_kv_len, \
                f"host_slot_mapping_fp8_fullkv/host_slot_mapping_scale_fullkv length mismatch: {len(fp8_flat_indices)} != total_kv_len={total_kv_len}"

            # Store extended mappings for indexer full KV gathering
            metadata.slot_mapping_fp8_fullkv = host_slot_mapping_fp8_fullkv.cuda(
                non_blocking=True)
            metadata.slot_mapping_scale_fullkv = host_slot_mapping_scale_fullkv.cuda(
                non_blocking=True)
        else:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale

    def _update_k_cache(self, k_fp8: torch.Tensor, k_scale: torch.Tensor,
                        metadata: DSAtrtllmAttentionMetadata) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
            return

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
            self.layer_idx)

        num_tokens = k_fp8.shape[0]

        # The C++ op reinterprets k_fp8 (FP8) and k_scale (float32) as raw
        # bytes internally and only reads the first num_tokens entries from
        # the slot mapping buffers, avoiding Python-side view/slice overhead.
        torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8, k_scale, k_cache,
                                                    metadata.slot_mapping_fp8,
                                                    metadata.slot_mapping_scale,
                                                    num_tokens)

    def _call_mqa_logits(self, q_fp8: torch.Tensor, k_fp8: torch.Tensor,
                         k_scale: torch.Tensor, weights: torch.Tensor,
                         cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor,
                         q_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Dispatch to fp8_mqa_logits or fp8_fp4_mqa_logits based on use_fp4.

        For FP4 the gather output is typed as FP8 for historical reasons;
        reinterpret the bytes as the expected int8 / int32 layouts. DeepGEMM
        asserts kv_sf is 1D in both modes, so flatten the scale here.
        """
        if self.use_fp4:
            k_fp4_bytes = k_fp8.view(torch.int8)
            k_scale_int32 = k_scale.view(torch.int32).reshape(-1)
            # q_scale arrives here as (chunk_tokens, n_heads, 1) — fused_cat_fp4
            # emits one int32 per (token, head) carrying four UE8M0 exponents,
            # pre_indexer_proj reshapes back to (N, n_heads, 1), and
            # sparse_attn_indexer chunk-slices on axis 0. The DeepGEMM FP4
            # kernel asserts q_sf is 2D, so collapse the trailing unit axis.
            q_scale_2d = q_scale.reshape(-1, self.n_heads)
            return fp8_fp4_mqa_logits(
                (q_fp8, q_scale_2d),
                (k_fp4_bytes, k_scale_int32),
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
        return fp8_mqa_logits(q_fp8, (k_fp8, k_scale.reshape(-1)), weights,
                              cu_seqlen_ks, cu_seqlen_ke)

    def _call_paged_mqa_logits(self, q_decode: torch.Tensor,
                               k_cache: torch.Tensor,
                               weights_decode: torch.Tensor,
                               context_lens: torch.Tensor,
                               block_table: torch.Tensor,
                               scheduler_metadata_buffer: torch.Tensor,
                               max_seq_len: int,
                               q_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Dispatch to fp8_paged_mqa_logits or fp8_fp4_paged_mqa_logits."""
        if self.use_fp4:
            return fp8_fp4_paged_mqa_logits(
                (q_decode, q_scale), k_cache, weights_decode, context_lens,
                block_table, scheduler_metadata_buffer, max_seq_len)
        return fp8_paged_mqa_logits(q_decode, k_cache, weights_decode,
                                    context_lens, block_table,
                                    scheduler_metadata_buffer, max_seq_len)

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        use_custom_topk: bool = True,
        q_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for both prefill and decode phases.

        q_scale is only consumed by the FP4 dispatch; FP8 path ignores it.
        """
        # DSACacheManager hardcodes quant_block_size=128 (see its __init__);
        # FP4 uses per-block-32 UE8M0 scales but packs four of them into one
        # int32 so the cache-layout contribution is the same 4 bytes/token as
        # FP8 per-block-128 at head_dim=128.
        assert metadata.kv_cache_manager is None or \
            metadata.kv_cache_manager.quant_block_size == 128, \
            f"Unexpected quant_block_size {metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        # Update the indexer k cache before prefill chunks gather from it.
        self._update_k_cache(k_fp8, k_scale, metadata)

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        has_decode = num_generations > 0
        has_prefill = num_contexts > 0
        num_gen_tokens = num_tokens - num_ctx_tokens

        topk_indices_buffer = torch.empty(
            (hidden_states.shape[0], self.index_topk),
            dtype=torch.int32,
            device=hidden_states.device)
        if not use_custom_topk:
            topk_indices_buffer[:hidden_states.shape[0]] = -1

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:

                # Default to 8192 if sparse_attention_config is not available (e.g., in unit tests)
                q_split_threshold = metadata.sparse_attention_config.q_split_threshold if metadata.sparse_attention_config is not None else 8192
                q_split_eligible = q_split_threshold >= 0 and metadata.mapping is not None and not metadata.mapping.enable_attention_dp and metadata.mapping.tp_size > 1

                if q_split_eligible:
                    tp_rank = metadata.mapping.tp_rank
                    tp_size = metadata.mapping.tp_size

                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                    self.layer_idx)

                gather_head_dim = self.head_dim // 2 if self.use_fp4 else self.head_dim
                for chunk in metadata.indexer_prefill_chunks:
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
                        k_cache_4d, metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv, chunk.k_token_start,
                        num_k_tokens, gather_head_dim)

                    chunk_num_token = chunk.token_end - chunk.token_start
                    apply_q_split = q_split_eligible and chunk_num_token >= q_split_threshold
                    if apply_q_split:
                        chunk_q_start = chunk_num_token * tp_rank // tp_size
                        chunk_q_end = chunk_num_token * (tp_rank + 1) // tp_size
                    else:
                        chunk_q_start = 0
                        chunk_q_end = chunk_num_token

                    global_q_start = chunk.token_start + chunk_q_start
                    global_q_end = chunk.token_start + chunk_q_end

                    chunk_q_scale = q_scale[global_q_start:global_q_end,
                                            ...] if self.use_fp4 else None
                    logits = self._call_mqa_logits(
                        q_fp8[global_q_start:global_q_end, ...],
                        chunk_k_fp8,
                        chunk_k_scale,
                        weights[global_q_start:global_q_end, ...],
                        chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                        chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end],
                        chunk_q_scale,
                    )
                    if use_custom_topk:
                        torch.ops.trtllm.indexer_topk_prefill(
                            logits,
                            chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                            chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end],
                            topk_indices_buffer[global_q_start:global_q_end, :])
                    else:
                        topk_indices = logits.topk(min(self.index_topk,
                                                       logits.shape[-1]),
                                                   dim=-1)[1]
                        topk_indices -= chunk.cu_seqlen_ks[
                            chunk_q_start:chunk_q_end][:, None]

                        mask_lo = topk_indices >= 0
                        mask_hi = topk_indices - (
                            chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end] -
                            chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end]
                        )[:, None] < 0
                        mask = mask_lo & mask_hi

                        # local indices per sequence
                        topk_indices = topk_indices.masked_fill(~mask, -1)

                        topk_indices_buffer[
                            global_q_start:global_q_end, :topk_indices.
                            shape[-1]] = topk_indices.to(dtype=torch.int32)

                    if apply_q_split:
                        q_sizes = [(r + 1) * chunk_num_token // tp_size -
                                   r * chunk_num_token // tp_size
                                   for r in range(tp_size)]
                        topk_indices_buffer[
                            chunk.token_start:chunk.token_end, :] = allgather(
                                topk_indices_buffer[
                                    global_q_start:global_q_end, :],
                                metadata.mapping,
                                dim=0,
                                sizes=q_sizes)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = q_scale[:num_ctx_tokens,
                                      ...] if self.use_fp4 else None
                logits = self._call_mqa_logits(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_tokens, ...],
                    k_scale[:num_ctx_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    ctx_q_scale,
                )
                if use_custom_topk:
                    torch.ops.trtllm.indexer_topk_prefill(
                        logits, cu_seqlen_ks, cu_seqlen_ke,
                        topk_indices_buffer[:num_ctx_tokens, :])
                else:
                    topk_indices = logits.topk(min(self.index_topk,
                                                   logits.shape[-1]),
                                               dim=-1)[1]
                    topk_indices -= cu_seqlen_ks[:, None]
                    mask_lo = topk_indices >= 0
                    mask_hi = topk_indices - (cu_seqlen_ke -
                                              cu_seqlen_ks)[:, None] < 0
                    mask = mask_lo & mask_hi

                    # local indices per sequence
                    topk_indices = topk_indices.masked_fill(~mask, -1)
                    topk_indices_buffer[:num_ctx_tokens, :topk_indices.
                                        shape[-1]] = topk_indices.to(
                                            dtype=torch.int32)
        elif has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = \
                metadata.topk_indices_buffer[:num_ctx_tokens, :]

        if has_decode and not metadata.skip_indexer_for_gen_reqs:
            max_seq_len = metadata.kv_cache_manager.max_seq_len
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts:num_contexts +
                                             num_generations]
            max_decode_len = gen_seq_lens.max().item()
            min_decode_len = gen_seq_lens.min().item()
            assert max_decode_len == min_decode_len, "max_decode_len != min_decode_len, we need padding"

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[num_ctx_tokens:num_ctx_tokens + num_gen_tokens,
                             ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations
            # Because fp8_paged_mqa_logits can only support next_n == 1/2/4 on sm100, and
            # next_n == 1/2 on sm90, for other next_n, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if not metadata.use_expanded_buffers_for_mtp or next_n == 1:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                # 2D context_lens slice from the pre-allocated buffer; matches
                # q_decode's (batch, next_n) layout required by the new
                # DeepGEMM paged MQA logits API.
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :
                                                        next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts:num_contexts + num_generations]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(
                    -1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[num_ctx_tokens:num_ctx_tokens +
                                     num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits with prepared decode metadata
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                self.layer_idx)

            decode_q_scale = q_scale[num_ctx_tokens:num_ctx_tokens +
                                     num_gen_tokens,
                                     ...] if self.use_fp4 else None
            if self.use_fp4:
                # q_decode shape is either (num_generations, next_n, n_heads,
                # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                decode_q_scale = decode_q_scale.view(q_decode.shape[0],
                                                     q_decode.shape[1],
                                                     self.n_heads)
            logits_decode = self._call_paged_mqa_logits(
                q_decode, k_cache, weights_decode, context_lens, block_table,
                scheduler_metadata_buffer, max_seq_len, decode_q_scale)

            if use_custom_topk:
                # Kernel expects kv_lens (total cache length), not seq_lens (new tokens)
                # This is because rowEnd = seq_len - next_n + offset + 1
                gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]

                pre_idx = None
                heuristic_scratch = None
                if self._enable_heuristic_topk:
                    local_layer = metadata.kv_cache_manager.layer_offsets[
                        self.layer_idx]
                    # Pass prev_topk directly; the +1 temporal offset is
                    # handled inside the C++ kernel (preIdxOffset += 1).
                    pre_idx = metadata.heuristic_prev_topk[
                        local_layer, :num_generations]
                    heuristic_scratch = \
                        metadata.heuristic_scratch_values[
                            :num_gen_tokens]

                # CuTE DSL top-k allocates O(num_gen_tokens * kv_len) global
                # memory. Beyond 256 tokens the extra memory becomes significant,
                # so we cap it at 256 for now and fall back to the CUDA C++
                # indexer_topk_decode. This limit can be removed if GPU memory
                # is not a bottleneck.
                if self.use_cute_dsl_topk and num_gen_tokens <= 256:
                    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                        logits_decode, gen_kv_lens_cuda,
                        topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :], self.index_topk,
                        next_n)
                else:
                    torch.ops.trtllm.indexer_topk_decode(
                        logits_decode,
                        gen_kv_lens_cuda,
                        topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :],
                        next_n,
                        self.index_topk,
                        pre_idx=pre_idx,
                        heuristic_scratch=heuristic_scratch)
            else:
                # padded
                positions = torch.arange(
                    max_seq_len, device=q_decode.device).unsqueeze(0).expand(
                        num_gen_tokens, -1)
                row_indices = torch.arange(num_gen_tokens,
                                           device=q_decode.device) // next_n
                next_n_offset = torch.arange(num_gen_tokens,
                                             device=q_decode.device) % next_n
                index_end_pos = (
                    metadata.kv_lens_cuda_runtime[num_contexts + row_indices] -
                    next_n + next_n_offset).unsqueeze(1)
                # index_end_pos: [B * N, 1]
                mask = positions <= index_end_pos
                # mask: [B * N, L]
                logits_decode = logits_decode.masked_fill(~mask, float('-inf'))
                topk_indices_decode = logits_decode.topk(
                    min(self.index_topk, logits_decode.shape[-1]),
                    dim=-1)[1].to(torch.int32)  # [B * N, K]
                # ensure we don't set indices for the top k
                # that is out of range(masked already)
                # this will happen if context length is shorter than K
                mask_decode = topk_indices_decode <= index_end_pos

                # local indices per sequence
                topk_indices_decode = topk_indices_decode.masked_fill(
                    ~mask_decode, -1)
                # Store in buffer
                topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                    num_gen_tokens, :topk_indices_decode.
                                    shape[-1]] = topk_indices_decode.to(
                                        dtype=torch.int32)

            if self._enable_heuristic_topk:
                local_layer = metadata.kv_cache_manager.layer_offsets[
                    self.layer_idx]
                decode_topk = topk_indices_buffer[
                    num_ctx_tokens:num_ctx_tokens + num_gen_tokens]
                last_mtp_topk = decode_topk[next_n - 1::next_n]
                metadata.heuristic_prev_topk[
                    local_layer, :num_generations].copy_(last_mtp_topk)

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[num_ctx_tokens:num_tokens, :] = \
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
        return topk_indices_buffer

    def _weight_scale(self, weights: torch.Tensor,
                      q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def _qk_projection_and_rope(self, qr: torch.Tensor, indexer_k: torch.Tensor,
                                position_ids: torch.Tensor):
        """Project Q/K and apply RoPE"""
        q = self.wq_b(qr)
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
        k_pe = k_pe[:, 0, :]
        return q_pe, q_nope, k_pe, k_nope

    def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
        """Concatenate and quantize for Q or K.

        FP8 mode: fused cat + FP8 quantize via CUDA kernel.
        FP4 mode: fused cat + per-block-32 FP4 E2M1 quantize via CUDA kernel.
        The returned packed bytes are int8 (two FP4 codes per byte) and the
        scale is int32 (four UE8M0 exponents packed little-endian).
        """
        if self.use_fp4:
            return torch.ops.trtllm.fused_cat_fp4(qk_pe, qk_nope)
        fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(
            qk_pe, qk_nope, self.scale_fmt == "ue8m0")
        return fp8_out, scale

    def pre_indexer_proj(
        self, qr: torch.Tensor, hidden_states: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs cublas_mm, qk_projection_and_rope, FP8 quantize, and weight
        scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The last
        tensor is only consumed by the FP4 kernel dispatch; the FP8 path
        ignores it. It is returned unconditionally so the two-op CUDA graph
        split in MLA.forward_dsa_proj sees a stable signature.
        """
        assert self._fused_wk_wp_weight is not None, \
            "post_load_weights() must be called before forward()"
        hidden_float = _to_float(hidden_states)
        with _tf32_matmul_enabled():
            # F.linear computes input @ weight.T internally; no explicit .t() needed.
            # _fused_wk_wp_weight is [head_dim + n_heads, hidden_size] (nn.Linear convention).
            # Goes through PyTorch's cuBLAS handle which respects allow_tf32 and
            # dispatches CUBLAS_COMPUTE_32F_FAST_TF32, unlike torch.ops.trtllm.cublas_mm
            # which uses its own handle and always falls back to CUDA-core SGEMM.
            fused_out = F.linear(hidden_float, self._fused_wk_wp_weight)
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads],
                                             dim=-1)
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE, FP8 quantize)
        indexer_k = indexer_k.to(hidden_states.dtype)

        q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(
            qr, indexer_k, position_ids)
        q, k = maybe_execute_in_parallel(
            lambda: self._prep_q_or_k(q_pe, q_nope),
            lambda: self._prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_fp8, q_scale = q
        k_fp8, k_scale = k
        if self.use_fp4:
            # FP4 packs two codes per byte, so the trailing dim is head_dim // 2.
            # fused_cat_fp4 flattens the leading dims to M=N*n_heads; restore
            # the (N, n_heads, ...) shape so downstream slicing in
            # sparse_attn_indexer (which indexes by token) lines up with
            # q_fp8. The DeepGEMM FP4 kernel applies the per-block q_scale
            # internally, so weights carry only softmax_scale * n_heads^-0.5.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = weights * self.weight_scale_factor
        else:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = self._weight_scale(weights, q_scale)

        return q_fp8, k_fp8, k_scale, weights, q_scale

    @torch.inference_mode()
    def forward(self, qr: torch.Tensor, hidden_states: torch.Tensor,
                metadata: DSAtrtllmAttentionMetadata,
                position_ids: torch.Tensor):
        q_fp8, k_fp8, k_scale, weights, q_scale = self.pre_indexer_proj(
            qr, hidden_states, position_ids)

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.sparse_attn_indexer(metadata,
                                        hidden_states,
                                        q_fp8,
                                        k_fp8,
                                        k_scale,
                                        weights,
                                        q_scale=q_scale)


class DSATrtllmAttention(TrtllmAttention):
    """TRT-LLM attention layer with DSA sparse indexer for MLA models."""

    Metadata = DSAtrtllmAttentionMetadata

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config: Optional[QuantConfig] = None,
            q_scaling: Optional[float] = None,
            pos_embd_params: Optional[PositionalEmbeddingParams] = None,
            mla_params: Optional[MLAParams] = None,
            skip_create_weights_in_init: bool = False,
            attention_chunk_size: Optional[int] = None,
            sparse_attention_config: Optional["SparseAttentionConfig"] = None,
            dtype: Optional[torch.dtype] = None,
            aux_stream: Optional[torch.cuda.Stream] = None,
            **kwargs):
        """Initialize DSA attention with an Indexer sub-module for sparse TopK selection."""
        if sparse_attention_config is None:
            raise ValueError(
                "sparse_attention_config is required for DSATrtllmAttention and cannot be None"
            )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs)

        self.indexer = Indexer(quant_config, pos_embd_params, mla_params,
                               skip_create_weights_in_init,
                               sparse_attention_config, dtype, layer_idx,
                               aux_stream)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Transform local TopK indices to global paged KV cache indices."""
        # Transform the local topk indices to global topk indices in paged kv cache
        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            forward_args.topk_indices, metadata,
            self.get_local_layer_idx(metadata), forward_args.is_generation)

        # TODO: Use sparse_attn_indexer to predict the indices for DSA attention
        # return self.indexer(q, k, metadata, hidden_states, qr, position_ids)
        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """No-op KV prediction; DSA uses indexer-based selection instead."""
        return None, None

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
        """Apply RoPE, append latent cache to paged KV, and assign query for MLA."""
        if is_generation:
            cached_token_indptr = metadata.gen_cached_token_indptr
            kv_indptr = metadata.gen_kv_indptr
            num_seqs = metadata.num_generations
            max_seq_len = metadata.max_gen_seq_len
            block_offsets = metadata.kv_cache_block_offsets[:, metadata.
                                                            num_contexts:]
        else:
            cached_token_indptr = metadata.ctx_cached_token_indptr
            kv_indptr = metadata.ctx_kv_indptr
            num_seqs = metadata.num_contexts
            max_seq_len = metadata.max_ctx_seq_len
            block_offsets = metadata.kv_cache_block_offsets
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        sink_token_length = 0
        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_seqs,
            cached_token_indptr,
            kv_indptr,
            max_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
            beam_width,
            self.quant_mode,
        )


class DSACacheManager(KVCacheManager):
    """KV cache manager for DSA with additional indexer K-cache pools."""

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: "SparseAttentionConfig",
        **kwargs,
    ) -> None:
        """Initialize cache manager with indexer K-cache pool per layer."""
        self.quant_block_size = 128
        self.index_head_dim = sparse_attn_config.index_head_dim
        # FP4 mode packs the indexer K cache as head_dim/2 data bytes + 4
        # scale bytes (vs. head_dim + 4 for FP8). The C++ WindowBlockManager
        # allocates the pool with this smaller stride when the flag is set.
        self.use_fp4 = sparse_attn_config.indexer_k_dtype == "fp4"

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            enable_indexer_k_cache=True,
            indexer_k_cache_quant_block_size=128,
            indexer_k_cache_index_head_dim=self.index_head_dim,
            indexer_k_cache_use_fp4=self.use_fp4,
            **kwargs,
        )
        self.num_blocks = self.blocks_in_primary_pool

        # Indexer K cache pool for DSA attention
        # Shape: [num_blocks, self.tokens_per_block * (index_head_dim + scale_size)]
        # Non-interleaved layout: [fp8_tok0 | fp8_tok1 | ... | scale_tok0 | scale_tok1 | ...]
        # Store FP8-quantized k values from the indexer
        self.indexer_k_cache_pool_per_layer = [
            self.get_indexer_k_cache_pool_data(layer_idx)
            for layer_idx in range(self.num_local_layers)
        ]

    def get_indexer_k_cache_buffers(self, layer_idx: int):
        """Get indexer k cache buffer from a specific layer pool."""
        block_size = self.tokens_per_block
        data_bytes = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        per_token_size = data_bytes + self.index_head_dim // self.quant_block_size * 4
        layer_offset = self.layer_offsets[layer_idx]
        return self.indexer_k_cache_pool_per_layer[layer_offset].view(
            self.num_blocks, block_size, 1, per_token_size)

    def shutdown(self):
        """Release indexer K-cache pool references before C++ buffer cleanup."""
        # Clear Python references BEFORE C++ frees the underlying CUDA buffers
        self.indexer_k_cache_pool_per_layer = []
        super().shutdown()

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig,
                                 mapping: Mapping,
                                 num_layers: Optional[int] = None,
                                 **kwargs):
        """Estimate total cache bytes per token including indexer K-cache overhead."""
        config = model_config.pretrained_config
        sparse_attn_config = model_config.sparse_attention_config
        index_head_dim = sparse_attn_config.index_head_dim
        quant_block_size = 128
        # Under FP4 the indexer stores two E2M1 codes per byte, so the
        # per-token data footprint halves (132 B -> 68 B at index_head_dim=128);
        # the scale bytes are unchanged (4 per token, one int32 holding four
        # UE8M0 exponents at quant_block_size=32 after packing).
        use_fp4 = sparse_attn_config.indexer_k_dtype == "fp4"
        indexer_data_dim = index_head_dim // 2 if use_fp4 else index_head_dim

        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get head dim
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers)
        mem_per_token *= num_attention_layers * head_dim

        # 1 for K, others for indexer K cache
        head_dim_factor = (indexer_data_dim +
                           index_head_dim // quant_block_size * 4) / head_dim
        kv_factor = 1 + head_dim_factor
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        """Compute actual cache bytes per token from instance configuration."""
        # self.kv_factor for K, others for indexer K cache.
        # Under FP4 the indexer data portion is halved (two E2M1 codes per
        # byte); scale bytes are unchanged.
        indexer_data_dim = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        head_dim_factor = (indexer_data_dim + self.index_head_dim //
                           self.quant_block_size * 4) / self.head_dim
        kv_factor = self.kv_factor + head_dim_factor
        cache_size_per_token = math.ceil(
            kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim)

        if self.dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                              DataType.FLOAT, DataType.NVFP4):
            raise ValueError(f'Cannot support {self.dtype} KV cache.')

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
        return cache_size_bytes_per_token
''',
    'key-files/tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py': r'''import threading
from typing import List, Optional, Set, Tuple

import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from ..._utils import get_sm_version
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

# Idempotency guard for warmup_heuristic_topk_decode — keyed by
# (device_index, top_k, hint_size, num_cols). Prevents repeated allocations
# and synchronizations when multiple Indexer modules invoke the warmup with
# the same parameters during model construction.
_HEURISTIC_TOPK_WARMUP_DONE: Set[Tuple[int, int, int, int]] = set()
_HEURISTIC_TOPK_WARMUP_LOCK = threading.Lock()

if IS_CUTLASS_DSL_AVAILABLE:
    from .cute_dsl_custom_ops import GroupedGemmInputsHelper


def _register_fake():

    @torch.library.register_fake("trtllm::allreduce")
    def allreduce(
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        strategy: int,
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
    ) -> List[torch.Tensor]:
        from tensorrt_llm.functional import AllReduceFusionOp
        if op == int(AllReduceFusionOp.NONE):
            return [torch.empty_like(input)]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM):
            norm_out = torch.empty_like(input)
            residual_out = torch.empty_like(input)
            return [norm_out, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8):
            quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
            residual_out = torch.empty_like(input)
            return [quant_out, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8):
            norm_out = torch.empty_like(input)
            quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
            residual_out = torch.empty_like(input)
            return [norm_out, quant_out, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4):
            fp4_shape, scale_shape = fp4_utils.get_fp4_shape(input.shape, 16)
            quant_fp4 = input.new_empty(fp4_shape, dtype=torch.uint8)
            scale_fp4 = input.new_empty(scale_shape, dtype=torch.uint8)
            residual_out = torch.empty_like(input)
            return [quant_fp4, scale_fp4, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4):
            fp4_shape, scale_shape = fp4_utils.get_fp4_shape(input.shape, 16)
            quant_fp4 = input.new_empty(fp4_shape, dtype=torch.uint8)
            scale_fp4 = input.new_empty(scale_shape, dtype=torch.uint8)
            norm_out = torch.empty_like(input)
            residual_out = torch.empty_like(input)
            return [norm_out, quant_fp4, scale_fp4, residual_out]
        else:
            return [torch.empty_like(input)]

    @torch.library.register_fake("trtllm::allreduce_pg")
    def _(
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        rank: int,
        pg,
        strategy: int,
        op: int,
        eps: float,
        trigger_completion_at_end: bool,
    ):
        return allreduce(input, residual, norm_weight, scale, bias, workspace,
                         group, strategy, op, eps, trigger_completion_at_end)

    # MNNVL Allreduce
    @torch.library.register_fake("trtllm::mnnvl_fusion_allreduce")
    def _(input, residual, gamma, epsilon, buffer, buffer_flags,
          rmsnorm_fusion):
        output = input.new_empty(input.shape)
        if rmsnorm_fusion:
            residual_out = residual.new_empty(residual.shape)
        else:
            residual_out = None
        return [output, residual_out]

    @torch.library.register_fake("trtllm::moe_allreduce")
    def _(residual, norm_weight, device_num_experts, scale_input,
          active_experts_token_input, token_input, workspace, rank, nranks,
          eps):
        norm_out = torch.empty_like(token_input)
        residual_out = torch.empty_like(residual)
        return [norm_out, residual_out]

    @torch.library.register_fake("trtllm::minimax_allreduce_rms")
    def _(input, norm_weight, workspace, rank, nranks, eps,
          trigger_completion_at_end):
        return torch.empty_like(input)

    @torch.library.register_fake("trtllm::minimax_allreduce_rms_qk")
    def _(q, k, norm_weight_q, norm_weight_k, workspace, rank, nranks, eps,
          trigger_completion_at_end):
        return [torch.empty_like(q), torch.empty_like(k)]

    @torch.library.register_fake("trtllm::allgather")
    def allgather(input, sizes, group):
        if sizes is None:
            output_shape = (len(group) * input.shape[0], *input.shape[1:])
        else:
            output_shape = (sum(sizes), *input.shape[1:])
        return input.new_empty(output_shape)

    @torch.library.register_fake("trtllm::allgather_pg")
    def _(input, sizes, group, process_group):
        return allgather(input, sizes, group)

    @torch.library.register_fake("trtllm::cublas_scaled_mm")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias,
        out_dtype,
        output_buffer_kind: int = 0,
        group: Optional[List[int]] = None,
    ):
        shape = [i for i in mat_a.shape]
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(shape, dtype=out_dtype)
        return ret

    @torch.library.register_fake("trtllm::cuda_scaled_mm")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        output_buffer_kind: int = 0,
        group: Optional[List[int]] = None,
    ):
        shape = [i for i in mat_a.shape]
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(shape, dtype=out_dtype)
        return ret

    @torch.library.register_fake("trtllm::cublas_mm")
    def _(mat_a,
          mat_b,
          bias,
          out_dtype,
          output_buffer_kind: int = 0,
          group: Optional[List[int]] = None):
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(
            shape, dtype=out_dtype if out_dtype is not None else mat_a.dtype)
        return ret

    @torch.library.register_fake("trtllm::dsv3_router_gemm_op")
    def _(mat_a, mat_b, bias, out_dtype):
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(
            shape, dtype=out_dtype if out_dtype is not None else mat_a.dtype)
        return ret

    @torch.library.register_fake("trtllm::dsv3_fused_a_gemm_op")
    def _(mat_a, mat_b, bias, out_dtype):
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(
            shape, dtype=out_dtype if out_dtype is not None else mat_a.dtype)
        return ret

    @torch.library.register_fake("trtllm::fp4_gemm")
    def _(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        mat1_scale: torch.Tensor,
        mat2_scale: torch.Tensor,
        global_scale: torch.Tensor,
        sf_use_ue8m0: bool,
        out_dtype=None,
    ):
        shape = list(mat1.shape)
        shape[-1] = mat2.shape[0]
        ret = mat1.new_empty(shape, dtype=out_dtype)
        return ret

    @torch.library.register_fake("trtllm::noaux_tc_op")
    def _(scores, scores_with_bias, n_group, topk_group, topk,
          routed_scaling_factor):
        shape = list(scores.shape)
        shape[-1] = topk
        return scores.new_empty(shape,
                                dtype=scores_with_bias.dtype), scores.new_empty(
                                    shape, dtype=torch.int32)

    @torch.library.register_fake("trtllm::indexer_topk_prefill")
    def _(logits, row_starts, row_ends, indices, index_topk):
        # In-place operation, no return value (void function)
        pass

    @torch.library.register_fake("trtllm::indexer_topk_decode")
    def _(logits,
          seq_lens,
          indices,
          next_n,
          index_topk,
          pre_idx=None,
          heuristic_scratch=None):
        # In-place operation, no return value (void function)
        pass

    @torch.library.register_fake("trtllm::userbuffers_allreduce_finalize")
    def _(input, force_applying_finalize):
        return torch.empty_like(input)

    @torch.library.register_fake("trtllm::fp8_block_scaling_gemm_impl")
    def _(a, b, a_scale, b_scale):
        m = a.shape[0]
        n = b.shape[0]
        return a.new_empty((m, n), dtype=torch.bfloat16)

    @torch.library.register_fake("tensorrt_llm::quantize_e4m3_per_tensor")
    def _(input: torch.Tensor):
        scale_shape = [1] * input.dim()
        return (input.new_empty(input.shape, dtype=torch.float8_e4m3fn),
                input.new_empty(scale_shape, dtype=input.dtype))

    @torch.library.register_fake(
        "tensorrt_llm::static_quantize_e4m3_per_tensor")
    def _(input: torch.Tensor, scale: torch.Tensor):
        return torch.empty_like(input, dtype=torch.float8_e4m3fn), scale.clone()

    @torch.library.register_fake("trtllm::fp4_quantize")
    def _(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        sf_vec_size: int,
        sf_use_ue8m0=False,
        swizzled_layout=True,
    ):
        output_shape, scale_shape = fp4_utils.get_fp4_shape(
            input.shape, sf_vec_size, swizzled_layout)

        return (input.new_empty(output_shape, dtype=torch.uint8),
                global_scale.new_empty(scale_shape, dtype=torch.uint8))

    @torch.library.register_fake("trtllm::fp4_quantize_with_reorder_residual")
    def _(
        X: torch.Tensor,
        input_scale: torch.Tensor,
        reorder_index: torch.Tensor,
        KE: int,
        is_act: bool,
    ):
        M = X.size(0)
        KQ = X.size(1)
        K = KQ + KE

        # QX shape: [M, K/2]
        QX = X.new_empty((M, K // 2), dtype=torch.uint8)

        # SFX shape: swizzled layout size for scale factors
        # isSfSwizzledLayout = True, sf_vec_size = 16
        SFSize = fp4_utils.pad_up(M, 128) * fp4_utils.pad_up(K // 16, 4)
        SFX = X.new_empty((SFSize, ), dtype=torch.uint8)

        return QX, SFX

    @torch.library.register_fake("trtllm::fp4_quantize_with_residual")
    def _(
        X: torch.Tensor,
        input_scale: torch.Tensor,
        KE: int,
        is_act: bool,
    ):
        M = X.size(0)
        KQ = X.size(1)
        K = KQ + KE

        # QX shape: [M, K/2]
        QX = X.new_empty((M, K // 2), dtype=torch.uint8)

        # SFX shape: swizzled layout size for scale factors
        # isSfSwizzledLayout = True, sf_vec_size = 16
        SFSize = fp4_utils.pad_up(M, 128) * fp4_utils.pad_up(K // 16, 4)
        SFX = X.new_empty((SFSize, ), dtype=torch.uint8)

        return QX, SFX

    @torch.library.register_fake("trtllm::mxfp8_quantize")
    def _(
        input: torch.Tensor,
        swizzled_layout: bool = True,
        alignment: int = 32,
    ):
        SF_VEC_SIZE = 32

        def pad_up(x, m: int):
            return (x + m - 1) // m * m

        m_val = 1
        for d in input.shape[:-1]:
            m_val = m_val * d

        k = input.shape[-1]
        padded_k = pad_up(k, alignment)

        out_shape = list(input.shape)
        out_shape[-1] = padded_k

        # Output tensor: float8_e4m3fn, last dim padded to alignment
        val_mxfp8 = input.new_empty(out_shape, dtype=torch.float8_e4m3fn)

        # Scale tensor: 1D uint8, size depends on swizzled vs linear layout
        cols = padded_k // SF_VEC_SIZE
        if swizzled_layout:
            sf_size = pad_up(m_val, 128) * pad_up(cols, 4)
        else:
            sf_size = m_val * cols

        scale_fp8_sf = input.new_empty((sf_size, ), dtype=torch.uint8)
        return val_mxfp8, scale_fp8_sf

    @torch.library.register_fake("trtllm::mxe4m3_mxe2m1_block_scale_moe_runner")
    def _(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        valid_hidden_size: Optional[int],
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        out_hidden_size = valid_hidden_size if valid_hidden_size is not None else hidden_size

        if output is not None:
            return output

        return hidden_states.new_empty((num_tokens, out_hidden_size),
                                       dtype=torch.bfloat16)

    @torch.library.register_fake("trtllm::calculate_nvfp4_global_scale")
    def _(input: torch.Tensor, tokens_per_batch: Optional[torch.Tensor]):
        return input.new_empty((input.shape[:-1], 1), dtype=torch.float32)

    @torch.library.register_fake("trtllm::moe_comm")
    def _(
        inputs: List[torch.Tensor],
        send_rank_cum_sum: torch.Tensor,
        send_indices: torch.Tensor,
        recv_rank_cum_sum: torch.Tensor,
        recv_indices: torch.Tensor,
        all_workspaces: torch.Tensor,
        output_allocation_count: int,
        ep_rank: int,
        ep_size: int,
        need_zero_output: Optional[List[bool]],
    ):
        outputs = []
        for input_tensor in inputs:
            output_tensor = torch.empty(
                (output_allocation_count, input_tensor.shape[1]),
                dtype=input_tensor.dtype,
                device=input_tensor.device)
            outputs.append(output_tensor)
        return outputs

    @torch.library.register_fake("trtllm::moe_a2a_dispatch")
    def _(
        token_selected_experts: torch.Tensor,
        input_payloads: List[torch.Tensor],
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        num_experts: int,
        eplb_local_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int, torch.Tensor]:
        recv_tensors: List[torch.Tensor] = []
        for payload in input_payloads:
            elements_per_token = payload.shape[1]
            recv_tensors.append(
                payload.new_empty(
                    (ep_size, runtime_max_tokens_per_rank, elements_per_token)))

        if eplb_local_stats is None:
            eplb_gathered_stats = workspace.new_empty((0, ), dtype=torch.int32)
        else:
            eplb_gathered_stats = workspace.new_empty(
                (ep_size, eplb_local_stats.shape[0]), dtype=torch.int32)

        combine_payload_offset = 0
        return recv_tensors, combine_payload_offset, eplb_gathered_stats

    @torch.library.register_fake("trtllm::moe_a2a_combine")
    def _(
        payload: torch.Tensor,
        local_num_tokens: int,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        runtime_max_tokens_per_rank: int,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        combine_payload_offset: int,
        payload_in_workspace: bool,
        use_low_precision: bool = False,
    ) -> torch.Tensor:
        return payload.new_empty((local_num_tokens, payload.shape[2]))

    @torch.library.register_fake("trtllm::moe_a2a_initialize")
    def _(
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        max_num_tokens_per_rank: int,
        eplb_stats_num_experts: Optional[int] = None,
    ) -> torch.Tensor:
        return torch.empty((10, ), dtype=torch.int64, device="cpu")

    @torch.library.register_fake("trtllm::moe_a2a_sanitize_expert_ids")
    def _(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        ep_rank: int,
        invalid_expert_id: int,
    ) -> None:
        return None

    @torch.library.register_fake("trtllm::moe_a2a_get_combine_payload_tensor")
    def _(
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        runtime_max_tokens_per_rank: int,
        combine_payload_offset: int,
        out_dtype: torch.dtype,
        hidden_size: int,
    ) -> torch.Tensor:
        return workspace.new_empty(
            (ep_size * runtime_max_tokens_per_rank, hidden_size),
            dtype=out_dtype)

    @torch.library.register_fake("trtllm::get_moe_commworkspace_size_per_rank")
    def _(ep_size: int):
        return 0

    @torch.library.register_fake("trtllm::set_moe_max_usable_sm_count")
    def _(max_sm_count: int):
        pass

    @torch.library.register_fake("trtllm::moe_load_balance_wait_gpu_stage")
    def _(single_layer_load_balancer_ptr: int):
        return torch.empty((1, ),
                           dtype=torch.int32,
                           device=torch.device("cuda"))

    @torch.library.register_fake("trtllm::moe_load_balance_set_cpu_stage")
    def _(single_layer_load_balancer_ptr: int):
        pass

    @torch.library.register_fake("trtllm::moe_load_balance_statistic")
    def _(gathered_raw_expert_ids: torch.Tensor, enabled: torch.Tensor,
          single_layer_load_balancer_ptr: int, is_first_stage: bool,
          is_last_stage: bool):
        pass

    @torch.library.register_fake(
        "trtllm::moe_hierarchical_statistic_local_device")
    def _(local_raw_expert_ids: torch.Tensor,
          local_expert_token_count: torch.Tensor, enabled: torch.Tensor,
          single_layer_load_balancer_ptr: int, is_first_stage: bool,
          is_last_stage: bool):
        pass

    @torch.library.register_fake("trtllm::moe_hierarchical_statistic_update")
    def _(global_expert_token_count: torch.Tensor, enabled: torch.Tensor,
          single_layer_load_balancer_ptr: int):
        pass

    @torch.library.register_fake("trtllm::moe_load_balance_routing")
    def _(single_layer_load_balancer_ptr: int,
          token_selected_experts: torch.Tensor, offset_by_ep_rank: bool):
        return torch.empty_like(token_selected_experts)

    @torch.library.register_fake("trtllm::memset_expert_ids")
    def _(experts_ids: torch.Tensor, recv_rank_count_cumsum: torch.Tensor,
          max_token_count_per_rank: int, top_k: int, invalid_expert_id: int,
          ep_size: int):
        pass

    @torch.library.custom_op("trtllm::group_rms_norm_base",
                             mutates_args=("outputs", ))
    def group_rms_norm_base(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm_base.register_fake
    def _(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> List[torch.Tensor]:
        return outputs

    @torch.library.custom_op("trtllm::group_rms_norm_large_batch",
                             mutates_args=("outputs", ))
    def group_rms_norm_large_batch(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm_large_batch.register_fake
    def _(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> List[torch.Tensor]:
        return outputs

    # Use groupRMSNormHeuristic which automatically selects between regular and large batch kernels
    @torch.library.custom_op("trtllm::group_rms_norm_heuristic",
                             mutates_args=("outputs", ))
    def group_rms_norm_heuristic(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm_heuristic.register_fake
    def _(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> List[torch.Tensor]:
        return outputs

    @torch.library.register_fake(
        "trtllm::mtp_sampling_and_accepted_draft_tokens_op")
    def _(logits: torch.Tensor, draft_tokens: torch.Tensor,
          target_tokens: torch.Tensor, num_mtp_modules: int, batch_size: int,
          num_context_request: int, vocab_size: int):
        return logits.new_empty((batch_size, num_mtp_modules + 1),
                                dtype=torch.int32), logits.new_empty(
                                    (batch_size, ), dtype=torch.int32)

    @torch.library.register_fake("trtllm::fp8_quantize_1x128")
    def _(input: torch.Tensor, use_ue8m0: bool = False):
        pad_m = fp4_utils.pad_up(input.shape[0], 4)
        blocked_n = (input.shape[1] + 127) // 128
        if get_sm_version() >= 100:
            sz = (blocked_n, input.shape[0])
        else:
            sz = (fp4_utils.pad_up(pad_m * blocked_n * 4, 128) // 4, )
        return torch.empty_like(input,
                                dtype=torch.float8_e4m3fn), input.new_empty(
                                    sz, dtype=torch.float)

    @torch.library.register_fake("trtllm::fused_cat_fp8")
    def _(pe: torch.Tensor, nope: torch.Tensor, use_ue8m0: bool = False):
        pe_dim = pe.shape[-1]
        nope_dim = nope.shape[-1]
        head_dim = pe_dim + nope_dim
        M = pe.numel() // pe_dim
        fp8_out = pe.new_empty((M, head_dim), dtype=torch.float8_e4m3fn)
        scale_out = pe.new_empty((M, 1), dtype=torch.float32)
        return fp8_out, scale_out

    @torch.library.register_fake("trtllm::fused_cat_fp4")
    def _(pe: torch.Tensor, nope: torch.Tensor):
        pe_dim = pe.shape[-1]
        nope_dim = nope.shape[-1]
        head_dim = pe_dim + nope_dim
        M = pe.numel() // pe_dim
        packed = pe.new_empty((M, head_dim // 2), dtype=torch.int8)
        scale = pe.new_empty((M, 1), dtype=torch.int32)
        return packed, scale

    @torch.library.register_fake("trtllm::causal_conv1d_fwd")
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias_: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
    ) -> None:
        pass

    @torch.library.register_fake("trtllm::causal_conv1d_update")
    def _(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias_: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens_: Optional[torch.Tensor],
        conv_state_indices_: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> None:
        pass

    @torch.library.register_fake("trtllm::moe_permute_op")
    def _(
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc2_expert_weights: torch.Tensor,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor],
        num_experts_per_node: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        cluster_size: int,
        cluster_rank: int,
        min_latency_mode: bool,
        use_fp8_block_scaling: bool,
    ):

        experts_per_token = token_selected_experts.shape[1]
        num_rows = input.shape[0]
        hidden_size = input.shape[1]

        num_moe_inputs = experts_per_token * num_rows

        unpermuted_token_selected_experts_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        unpermuted_source_token_ids_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        permuted_source_token_ids_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        permuted_token_selected_experts_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        permuted_data_tensor = input.new_empty((num_moe_inputs, hidden_size),
                                               dtype=torch.float32)
        expert_first_token_offset_tensor = token_selected_experts.new_empty(
            (num_experts_per_node + 1, ), dtype=torch.int64)
        permuted_token_final_scales_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.float32)
        src_to_dest_map_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)

        return (
            unpermuted_token_selected_experts_tensor,
            unpermuted_source_token_ids_tensor,
            permuted_source_token_ids_tensor,
            permuted_token_selected_experts_tensor,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            permuted_token_final_scales_tensor,
            src_to_dest_map_tensor,
        )

    @torch.library.register_fake("trtllm::moe_finalize_scale_op")
    def _(
        gemm2_output: torch.Tensor,
        fc2_expert_biases: torch.Tensor,
        unpermuted_final_scales: torch.Tensor,
        unpermuted_row_to_permuted_row: torch.Tensor,
        expert_for_source_row: torch.Tensor,
        expert_first_token_offset_tensor: torch.Tensor,
        num_rows: torch.SymInt,
        hidden_size: torch.SymInt,
        unpadded_hidden_size: torch.SymInt,
        experts_per_token: int,
        num_experts_per_node: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
    ):
        num_rows_val = int(num_rows)
        unpadded_hidden_size_val = int(unpadded_hidden_size)
        return gemm2_output.new_empty((num_rows_val, unpadded_hidden_size_val),
                                      dtype=gemm2_output.dtype)

    if IS_CUTLASS_DSL_AVAILABLE:

        @torch.library.register_fake("trtllm::moe_topk_sort")
        def _(
            routing_logits: torch.Tensor,
            routing_bias: Optional[torch.Tensor],
            num_experts: int,
            top_k: int,
            n_group: Optional[int],
            topk_group: Optional[int],
            local_expert_offset: int,
            local_num_experts: int,
            routed_scaling_factor: Optional[float],
            tile_tokens_dim: int,
            routing_method_type: int,
        ) -> List[torch.Tensor]:
            helper = GroupedGemmInputsHelper(
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=local_num_experts,
                local_expert_offset=local_expert_offset,
                tile_size=tile_tokens_dim,
            )
            num_tokens = routing_logits.size(0)
            device = routing_logits.device
            routing_bias_dtype = torch.bfloat16 if routing_bias is None else routing_bias.dtype
            max_num_tiles = helper.get_max_num_tiles(num_tokens)
            max_num_permuted_tokens = helper.get_max_num_permuted_tokens(
                num_tokens)
            tile_idx_to_expert_idx = torch.empty((max_num_tiles, ),
                                                 dtype=torch.int32,
                                                 device=device)
            tile_idx_to_mn_limit = torch.empty((max_num_tiles, ),
                                               dtype=torch.int32,
                                               device=device)
            expanded_idx_to_permuted_idx = torch.empty((num_tokens, top_k),
                                                       dtype=torch.int32,
                                                       device=device)
            permuted_idx_to_expanded_idx = torch.empty(
                (max_num_permuted_tokens, ), dtype=torch.int32, device=device)
            total_num_padded_tokens = torch.empty((1, ),
                                                  dtype=torch.int32,
                                                  device=device)
            num_non_exiting_tiles = torch.empty((1, ),
                                                dtype=torch.int32,
                                                device=device)
            new_token_final_scales = torch.empty((num_tokens, top_k),
                                                 dtype=routing_bias_dtype,
                                                 device=device)
            return [
                tile_idx_to_expert_idx, tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
                total_num_padded_tokens, num_non_exiting_tiles,
                new_token_final_scales
            ]

        @torch.library.register_fake("trtllm::moe_sort")
        def _(
            token_selected_experts: torch.Tensor,
            token_final_scales: torch.Tensor,
            num_experts: int,
            top_k: int,
            local_expert_offset: int,
            local_num_experts: int,
            tile_tokens_dim: int,
        ) -> List[torch.Tensor]:
            helper = GroupedGemmInputsHelper(
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=local_num_experts,
                local_expert_offset=local_expert_offset,
                tile_size=tile_tokens_dim,
            )
            num_tokens = token_selected_experts.size(0)
            device = token_selected_experts.device
            max_num_tiles = helper.get_max_num_tiles(num_tokens)
            max_num_permuted_tokens = helper.get_max_num_permuted_tokens(
                num_tokens)
            tile_idx_to_expert_idx = torch.empty((max_num_tiles, ),
                                                 dtype=torch.int32,
                                                 device=device)
            tile_idx_to_mn_limit = torch.empty((max_num_tiles, ),
                                               dtype=torch.int32,
                                               device=device)
            expanded_idx_to_permuted_idx = torch.empty((num_tokens, top_k),
                                                       dtype=torch.int32,
                                                       device=device)
            permuted_idx_to_expanded_idx = torch.empty(
                (max_num_permuted_tokens, ), dtype=torch.int32, device=device)
            total_num_padded_tokens = torch.empty((1, ),
                                                  dtype=torch.int32,
                                                  device=device)
            num_non_exiting_tiles = torch.empty((1, ),
                                                dtype=torch.int32,
                                                device=device)
            return [
                tile_idx_to_expert_idx, tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
                total_num_padded_tokens, num_non_exiting_tiles
            ]

    @torch.library.register_fake("trtllm::moe_permute")
    def _(
        input: torch.Tensor,
        input_sf: Optional[torch.Tensor],
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0)
        permuted_output = torch.empty((max_num_permuted_tokens, input.size(1)),
                                      dtype=input.dtype,
                                      device=input.device)
        if input.dtype == torch.float4_e2m1fn_x2:
            hidden_size = input.size(1) * 2
            sf_vec_size = 16
            permuted_sf = torch.empty(
                (max_num_permuted_tokens * hidden_size // sf_vec_size, ),
                dtype=input_sf.dtype,
                device=input.device)
        else:
            permuted_sf = None
        return permuted_output, permuted_sf

    @torch.library.register_fake("trtllm::moe_unpermute")
    def _(
        permuted_input: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        topk_scales: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = expanded_idx_to_permuted_idx.size(0)
        output = torch.empty((num_tokens, permuted_input.size(1)),
                             dtype=permuted_input.dtype,
                             device=permuted_input.device)
        return output

    @torch.library.register_fake("trtllm::moe_swiglu")
    def _(
        input: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
    ) -> torch.Tensor:
        output = torch.empty((input.size(0), input.size(1) // 2),
                             dtype=input.dtype,
                             device=input.device)
        return output

    @torch.library.register_fake("trtllm::moe_swiglu_nvfp4_quantize")
    def _(
        input: torch.Tensor,
        global_sf: float,
        tile_idx_to_mn_limit: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_num_permuted_tokens = input.size(0)
        interm_size = input.size(1) // 2
        sf_vec_size = 16
        output = torch.empty((max_num_permuted_tokens, interm_size // 2),
                             dtype=torch.float4_e2m1fn_x2,
                             device=input.device)
        output_sf = torch.empty(
            (max_num_permuted_tokens * interm_size // sf_vec_size, ),
            dtype=torch.uint8,
            device=input.device)
        return output, output_sf

    @torch.library.register_fake("trtllm::moe_gelu")
    def _(
        input: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
    ) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.register_fake("trtllm::allgather_list")
    def allgather_list(input_list, sizes, group):
        assert len(input_list) > 0

        def create_output_tensor(i):
            shape = list(i.shape)
            if sizes is None:
                shape[0] *= len(group)
            else:
                shape[0] = sum(sizes)
            return i.new_empty(shape)

        return [create_output_tensor(i) for i in input_list]

    @torch.library.register_fake("trtllm::allgather_list_pg")
    def _(input_list, sizes, group, process_group):
        return allgather_list(input_list, sizes, group)

    @torch.library.register_fake("trtllm::reducescatter")
    def reducescatter(input, sizes, group):
        import tensorrt_llm
        local_rank = tensorrt_llm.mpi_rank()

        shape = list(input.shape)
        if sizes is None:
            shape[0] = shape[0] // len(group)
        else:
            shape[0] = sizes[local_rank]
        return input.new_empty(shape)

    @torch.library.register_fake("trtllm::reducescatter_pg")
    def _(input, sizes, group, process_group):
        return reducescatter(input, sizes, group)

    @torch.library.register_fake("trtllm::block_scale_interleave")
    def _(sf: torch.Tensor):
        rows = sf.shape[-2]
        cols = sf.shape[-1]
        expert_out_size = fp4_utils.pad_up(rows, 128) * fp4_utils.pad_up(
            cols, 4)
        num_experts = sf.shape[0] if len(sf.shape) == 3 else 1
        return sf.new_empty((num_experts * expert_out_size, ),
                            dtype=torch.uint8)

    @torch.library.register_fake("trtllm::block_scale_interleave_reverse")
    def _(sf: torch.Tensor):
        return torch.empty_like(sf, dtype=torch.uint8)

    @torch.library.register_fake("trtllm::moe_finalize_allreduce")
    def _(input, residual, norm_weight, expanded_idx_to_permuted_idx,
          shared_expert_output, expert_scale_factor, workspace, rank, nranks,
          eps) -> List[torch.Tensor]:
        return [
            torch.empty_like(residual),
            torch.empty_like(residual),
        ]

    @torch.library.register_fake("trtllm::renorm_moe_routing_op")
    def _(router_logits, topk, output_dtype: torch.dtype = None):
        num_tokens = router_logits.shape[0]
        sz = (num_tokens, topk)
        output_dtype = output_dtype or torch.float32
        return router_logits.new_empty(
            sz, dtype=torch.int32), router_logits.new_empty(sz,
                                                            dtype=output_dtype)

    @torch.library.register_fake("trtllm::default_moe_routing_op")
    def _(router_logits, topk, output_dtype: torch.dtype = None):
        num_tokens = router_logits.shape[0]
        sz = (num_tokens, topk)
        output_dtype = output_dtype or torch.float32
        return router_logits.new_empty(
            sz, dtype=torch.int32), router_logits.new_empty(sz,
                                                            dtype=output_dtype)

    @torch.library.register_fake("trtllm::alltoall_helix")
    def _(input_list, group, num_lists):
        num_ranks = len(group)
        len(input_list) // num_ranks
        return [
            input_list[i].new_empty((num_ranks, ) + i.shape)
            for i in range(0, len(input_list), num_ranks)
        ]

    @torch.library.register_fake("trtllm::alltoall_helix_native")
    def _(partial_o, softmax_stats, workspace, cp_rank, cp_size):
        # Returns outputs with same shapes as inputs
        return partial_o.new_empty(partial_o.shape), softmax_stats.new_empty(
            softmax_stats.shape)

    @torch.library.register_fake("trtllm::initialize_helix_workspace")
    def _(workspace, cp_rank, cp_size):
        # This op initializes workspace in-place and returns nothing
        return None

    @torch.library.register_fake("trtllm::helix_post_process")
    def _(gathered_o, gathered_stats, scale):
        return gathered_o.new_empty(*gathered_o.shape[1:])

    @torch.library.register_fake("trtllm::helix_post_process_native")
    def _(gathered_o, gathered_stats, scale, cp_dim):
        # Remove the dimension at cp_dim (context parallelism dimension)
        out_shape = list(gathered_o.shape)
        del out_shape[cp_dim]
        return gathered_o.new_empty(*out_shape)

    @torch.library.register_fake("trtllm::tinygemm2")
    def _(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        # input [M, K], weight [N, K], bias [N]
        # Output should be [M, N]
        m = input.shape[0]
        n = weight.shape[0]
        return input.new_empty((m, n), dtype=input.dtype)

    @torch.library.register_fake("trtllm::cuda_core_nvfp4_gemm")
    def _(mat_a: torch.Tensor,
          mat_b: torch.Tensor,
          scale_a: torch.Tensor,
          scale_b: torch.Tensor,
          alpha: torch.Tensor,
          bias: Optional[torch.Tensor],
          out_dtype: Optional[torch.dtype],
          output_buffer_kind: int = 0,
          group: Optional[List[int]] = None):
        # mat_a: [M, K/2], mat_b: [N, K/2]
        # Output should be [M, N] with dtype=out_dtype
        m = mat_a.shape[0]
        n = mat_b.shape[0]
        return mat_a.new_empty((m, n), dtype=out_dtype)

    @torch.library.register_fake("trtllm::mla_rope_generation")
    def _(
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        attention_window_size: int,
        sink_token_length: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ) -> None:
        # This is a fake implementation for shape inference
        # The actual operation modifies fused_q and q_pe in-place
        return None

    @torch.library.register_fake("trtllm::fused_add_rms_norm_quant")
    def _(
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        sf_scale: Optional[torch.Tensor],
        use_rms_norm: bool = True,
        eps: float = 1e-5,
        output_hp_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        m, n = input.shape
        # normed_output_fp4: [M, N/8] as int32 (8 FP4 values packed per int32)
        normed_output_fp4 = input.new_empty((m, n // 8), dtype=torch.int32)
        # output: [M, N] pre-norm output, same dtype as input
        output = input.new_empty((m, n), dtype=input.dtype)
        # sf_out: scale factors, swizzled layout
        sf_vec_size = 16
        sf_size = ((m + 127) // 128) * 128 * ((n // sf_vec_size + 3) // 4) * 4
        sf_out = input.new_empty((sf_size, ), dtype=torch.uint8)
        # high_precision_normed_output: [M, N] optional, only when output_hp_norm=True
        hp_output = input.new_empty(
            (m, n), dtype=input.dtype) if output_hp_norm else None
        return normed_output_fp4, output, sf_out, hp_output

    @torch.library.register_fake("trtllm::fused_gated_rmsnorm_quant")
    def _(
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        group_size: int,
        eps: float = 1e-5,
        sf_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, n = x.shape
        # y_fp4: [M, N/8] as int32 (8 FP4 values packed per int32)
        y_fp4 = x.new_empty((m, n // 8), dtype=torch.int32)
        # sf_out: scale factors in swizzled layout
        sf_vec_size = 16
        sf_size = ((m + 127) // 128) * 128 * ((n // sf_vec_size + 3) // 4) * 4
        sf_out = x.new_empty((sf_size, ), dtype=torch.uint8)
        return y_fp4, sf_out

    @torch.library.register_fake("trtllm::fused_relu2_quantize")
    def _(
        input: torch.Tensor,
        sf_scale: torch.Tensor,
        sf_vec_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: 2D tensor [M, N] (bf16 or fp16)
        # output_fp4: [M, N/2] (packed FP4 values, 2 values per byte)
        # output_sf: swizzled scale factors
        output_shape, scale_shape = fp4_utils.get_fp4_shape(
            input.shape, sf_vec_size, is_swizzled_layout=True)
        output_fp4 = input.new_empty(output_shape, dtype=torch.uint8)
        output_sf = input.new_empty((scale_shape, ), dtype=torch.uint8)
        return output_fp4, output_sf

    @torch.library.register_fake("trtllm::build_decoder_info")
    def _(seq_q_offsets, seq_kv_offsets, padding_offsets, tokens_info,
          encoder_padding_offsets, packed_mask_row_offsets,
          seq_cp_partial_offsets, attention_mask, seq_q_lengths, seq_kv_lengths,
          fmha_tile_counter, dequant_scale_qkv, quant_scale_o, fmha_bmm1_scale,
          fmha_bmm2_scale, rotary_embedding_inv_freq,
          rotary_embedding_inv_freq_cache, cp_size, separate_qkv_scales,
          fmha_host_bmm1_scale, batch_size, max_q_seq_length,
          max_encoder_q_seq_length, attention_window_size, sink_token_length,
          num_tokens, remove_padding, attention_mask_type,
          rotary_embedding_scale, rotary_embedding_base, rotary_embedding_dim,
          rotary_scaling_type, rotary_embedding_max_positions):
        return True

    @torch.library.register_fake("trtllm::convert_req_index_to_global")
    def _(req_id: torch.Tensor, block_table: torch.Tensor,
          token_indices: torch.Tensor, block_size: int, num_topk_tokens: int,
          stride_factor: int, layer_id: int) -> torch.Tensor:
        return torch.empty_like(token_indices)

    @torch.library.register_fake("trtllm::indexer_k_cache_gather_op")
    def _(k_cache: torch.Tensor, slot_mapping_fp8: torch.Tensor,
          slot_mapping_scale: torch.Tensor, k_token_start: int, num_tokens: int,
          head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # head_dim is payload bytes per token: 128 for FP8 or 64 for FP4
        # (two packed E2M1 codes per byte). k_fp8 holds raw gathered bytes
        # view-cast as float8_e4m3fn; the FP4 caller reinterprets as int8
        # with two packed values.
        k_fp8 = k_cache.new_empty([num_tokens, head_dim],
                                  dtype=torch.float8_e4m3fn)
        k_scale = k_cache.new_empty([num_tokens, 1], dtype=torch.float32)
        return k_fp8, k_scale

    @torch.library.register_fake("trtllm::allocate_output")
    def _(like: torch.Tensor,
          output_buffer_kind: int,
          group: Optional[List[int]],
          shape: Optional[List[int]] = None,
          out_dtype: Optional[torch.dtype] = None):
        out_shape = shape if shape is not None else list(like.shape)
        dtype = out_dtype if out_dtype is not None else like.dtype
        return like.new_empty(out_shape, dtype=dtype), output_buffer_kind


def warmup_heuristic_topk_decode(top_k: int = 2048,
                                 hint_size: int = 2048,
                                 num_cols: int = 4096) -> None:
    """Pre-initialize cached hardware attributes in the C++ Scheme X dispatcher.

    The dispatcher inside ``invokeIndexerTopKDecode`` lazily queries
    ``cudaDeviceGetAttribute`` for ``MultiProcessorCount`` and
    ``L2CacheSize`` on its first call. Those host-side queries must not
    be issued during ``cudaStreamBeginCapture / EndCapture``: the values
    captured there become frozen into the graph and cannot be refreshed
    across replays on a different device.

    This warmup issues one small heuristic decode call so the static
    caches are populated before any CUDA Graph capture begins. Must be
    called from the Indexer setup hook (``layer_idx == 0``) when
    ``enable_heuristic_topk`` is true.

    Repeated invocations with the same ``(device, top_k, hint_size,
    num_cols)`` key are short-circuited so that constructing many Indexer
    modules in the same process does not re-allocate scratch tensors or
    issue redundant synchronizations.
    """
    key = (torch.cuda.current_device(), top_k, hint_size, num_cols)
    with _HEURISTIC_TOPK_WARMUP_LOCK:
        if key in _HEURISTIC_TOPK_WARMUP_DONE:
            return
        _HEURISTIC_TOPK_WARMUP_DONE.add(key)

    device = torch.device("cuda")
    logits = torch.zeros((1, num_cols), dtype=torch.float32, device=device)
    seq_lens = torch.tensor([num_cols], dtype=torch.int32, device=device)
    indices = torch.empty((1, top_k), dtype=torch.int32, device=device)
    pre_idx = torch.zeros((1, hint_size), dtype=torch.int32, device=device)
    scratch = torch.empty((top_k, ), dtype=torch.float32, device=device)
    torch.ops.trtllm.indexer_topk_decode(logits,
                                         seq_lens,
                                         indices,
                                         1,
                                         top_k,
                                         pre_idx=pre_idx,
                                         heuristic_scratch=scratch)
    torch.cuda.synchronize()
''',
    'key-files/tensorrt_llm/_torch/modules/attention.py': r'''import functools
import math
import os
import weakref
from typing import List, Optional, Union, cast

import torch
from torch import nn

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import (get_sm_version, is_sm_100f, nvtx_range,
                                 nvtx_range_debug)
from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (AttentionForwardArgs, AttentionInputType,
                                 AttentionMetadata, FlashInferAttentionMetadata,
                                 TrtllmAttention, TrtllmAttentionMetadata)
from ..attention_backend.interface import (AttentionBackend, AttentionMask,
                                           CustomAttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.sparse.dsa import (
    DSAtrtllmAttentionMetadata, transform_local_topk_and_prepare_pool_view)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import (AllReduceParams, HelixAllToAllNative, alltoall_helix,
                           cp_allgather, reducescatter)
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import (Fp4QuantizedTensor, get_model_extra_attrs,
                     is_torch_compiling, maybe_compiled_cat,
                     maybe_compiled_copy_)
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding

# Import FlashMLA sparse attention kernel
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None


def extract_extra_attrs(layer_idx: str, attn_type: str):
    assert attn_type in ["mla", "attn"], "Invalid attention type"
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    if attn_type == "mla":
        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
    else:
        assert isinstance(
            metadata,
            FlashInferAttentionMetadata,
        ) or isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )

    attn_layers = extra_attrs.get(attn_type + "_layers", None)
    assert attn_layers is not None, "Attention layer is not registered"
    attn_layer_ref = attn_layers.get(layer_idx, None)
    assert attn_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    attn_layer = attn_layer_ref()

    if attn_type == "mla":
        assert isinstance(
            attn_layer,
            MLA), "MLA layer must be a subclass of MLA or an instance of MLA"
    elif attn_type == "attn":
        assert isinstance(
            attn_layer, Attention
        ), "Attention layer must be a subclass of Attention or an instance of Attention"

    return metadata, attn_layer


def create_attn_outputs_impl(q: torch.Tensor, attention_mask: str,
                             layer_idx: str) -> List[torch.Tensor]:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    return attn_layer.create_output(q, metadata, attention_mask)


@torch.library.custom_op("trtllm::create_attn_outputs", mutates_args=())
def create_attn_outputs(q: torch.Tensor, attention_mask: str,
                        layer_idx: str) -> List[torch.Tensor]:
    return create_attn_outputs_impl(q, attention_mask, layer_idx)


@create_attn_outputs.register_fake
def _(q, attention_mask, layer_idx):
    return create_attn_outputs_impl(q, attention_mask, layer_idx)


@torch.library.custom_op("trtllm::attn_custom_op_inplace",
                         mutates_args=("output", "output_sf"))
def attn_custom_op_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    attention_mask: str,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    attention_window_size: Optional[int],
    attention_mask_data: Optional[torch.Tensor],
    attention_sinks: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
) -> None:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    mask = PredefinedAttentionMask(
        attention_mask
    ) if attention_mask != CustomAttentionMask.CUSTOM else CustomAttentionMask(
        attention_mask)
    # NVFP4 output cannot be supported by torch compile for TRTLLM backend.
    attn_layer._attn_impl(q,
                          k,
                          v,
                          metadata,
                          mask,
                          mrope_rotary_cos_sin,
                          mrope_position_deltas,
                          attention_window_size,
                          attention_mask_data,
                          output=output,
                          output_sf=output_sf,
                          attention_sinks=attention_sinks)


def _helix_post_process(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping: Mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    aux_stream: Optional[torch.cuda.Stream] = None,
    ln_events: Optional[list] = None,
) -> torch.Tensor:
    """Helix CP post-processing: all-to-all exchange and combine partial
    attention outputs across CP ranks.

    This is shared by both MHA (Attention) and MLA modules.  The only
    dimension that differs between the two callers is *value_dim*
    (``head_dim`` for MHA, ``kv_lora_rank`` for MLA).

    When *aux_stream* and *ln_events* are provided the two
    ``.contiguous()`` calls in the FIFO-v1 path are overlapped on
    separate CUDA streams for better performance.
    """
    if mapping.cp_config.get("use_nccl_for_alltoall", True):
        # NCCL-based implementation using alltoall_helix.
        chunks = []
        for t in [partial_o, softmax_stats]:
            t = t.transpose(1, 0).contiguous()
            chunks.extend(torch.split(t, t.shape[0] // mapping.cp_size))
        gathered = alltoall_helix(chunks, mapping.cp_group)
        gathered = [t.transpose(1, 2).contiguous() for t in gathered]
        return torch.ops.trtllm.helix_post_process(gathered[0], gathered[1],
                                                   1.0)
    else:
        # FIFO-based implementation using MNNVL workspace.
        helix = HelixAllToAllNative.get(mapping)
        num_tokens = partial_o.shape[0]
        cp_size = mapping.cp_size
        fifo_version = mapping.cp_config.get("fifo_version", 2)

        if fifo_version == 1:

            def reshape_o():
                return partial_o.view(num_tokens, cp_size, num_heads_tp_cp,
                                      value_dim).transpose(1, 2).contiguous()

            def reshape_s():
                return softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp,
                                          2).transpose(1, 2).contiguous()

            if aux_stream is not None and ln_events is not None:
                partial_o, softmax_stats = maybe_execute_in_parallel(
                    reshape_o,
                    reshape_s,
                    ln_events[0],
                    ln_events[1],
                    aux_stream,
                )
            else:
                partial_o = reshape_o()
                softmax_stats = reshape_s()

            partial_o_out, softmax_stats_out = helix.alltoall_native(
                partial_o, softmax_stats)
            return torch.ops.trtllm.helix_post_process_native(
                partial_o_out, softmax_stats_out, 1.0, 2)
        else:
            partial_o = partial_o.view(num_tokens, cp_size,
                                       num_heads_tp_cp * value_dim)
            softmax_stats = softmax_stats.view(num_tokens, cp_size,
                                               num_heads_tp_cp * 2)
            partial_o_out, softmax_stats_out = helix.alltoall_native(
                partial_o, softmax_stats)
            gathered_o = partial_o_out.view(num_tokens, cp_size,
                                            num_heads_tp_cp, value_dim)
            gathered_stats = softmax_stats_out.view(num_tokens, cp_size,
                                                    num_heads_tp_cp, 2)
            return torch.ops.trtllm.helix_post_process_native(
                gathered_o, gathered_stats, 1.0, 1)


def _helix_cp_pad(tensor: torch.Tensor, num_tokens: int,
                  cp_size: int) -> tuple[torch.Tensor, int]:
    """Pad tensor along dim-0 so its length is divisible by cp_size."""
    chunk_size = math.ceil(num_tokens / cp_size)
    padded_size = chunk_size * cp_size
    if num_tokens < padded_size:
        tensor = torch.nn.functional.pad(tensor,
                                         (0, 0, 0, padded_size - num_tokens),
                                         mode="constant",
                                         value=0)
    return tensor, chunk_size


def _helix_cp_allgather_input(hidden_states: torch.Tensor,
                              attn_metadata: AttentionMetadata,
                              mapping: Mapping, layer_idx: int) -> torch.Tensor:
    """AllGather hidden states from CP group for layers after the first.

    The first layer already has the full input from the embedding.
    Subsequent layers need to undo the previous layer's reduce-scatter.
    """
    if (mapping.has_cp_helix() and mapping.enable_attention_dp
            and layer_idx > 0):
        hidden_states = cp_allgather(hidden_states, mapping, dim=0)
        hidden_states = hidden_states[:attn_metadata.num_tokens]
    return hidden_states


def _helix_cp_output_projection(
    o_proj: Linear,
    attn_output: torch.Tensor,
    attn_metadata: AttentionMetadata,
    all_reduce_params: Optional[AllReduceParams],
    mapping: Mapping,
    mapping_o: Mapping,
    layer_idx: int,
    lora_params: Optional[dict] = None,
) -> torch.Tensor:
    """Apply output projection with reduce-scatter when Helix CP+DP is active.

    Reduce-scatter sums partial sums across the CP group and scatters the
    result so each CP rank processes a distinct token chunk through the MLP.
    Falls back to the standard AllReduce path otherwise.
    """
    if mapping.has_cp_helix() and mapping.enable_attention_dp:
        attn_output = o_proj(
            attn_output,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            lora_params=lora_params,
            layer_idx=layer_idx)

        attn_output, _ = _helix_cp_pad(attn_output, attn_metadata.num_tokens,
                                       mapping.cp_size)
        attn_output = reducescatter(attn_output, mapping_o, dim=0)
    else:
        attn_output = o_proj(attn_output,
                             all_reduce_params=all_reduce_params,
                             lora_params=lora_params,
                             layer_idx=layer_idx)

    return attn_output


def maybe_slice_for_helix_cp(tensor: torch.Tensor,
                             attn_metadata: AttentionMetadata,
                             mapping_with_cp: Optional[Mapping],
                             layer_idx: int) -> torch.Tensor:
    """Slice a tensor to this CP rank's chunk after reduce-scatter.

    For the first decoder layer, the residual comes from the embedding and
    has not been through a prior reduce-scatter.  This function slices it
    so it aligns with the reduce-scattered attention output.  For
    subsequent layers the residual already has the correct size, so this
    is a no-op.

    Call this in the decoder layer on the residual *after* the attention
    forward, so that Attention/MLA forward signatures stay unchanged.
    """
    if (mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
            and mapping_with_cp.enable_attention_dp and layer_idx == 0):
        tensor, chunk_size = _helix_cp_pad(tensor, attn_metadata.num_tokens,
                                           mapping_with_cp.cp_size)
        start = mapping_with_cp.cp_rank * chunk_size
        tensor = tensor[start:start + chunk_size]
    return tensor


def maybe_allgather_for_helix_cp(
        hidden_states: torch.Tensor, attn_metadata: AttentionMetadata,
        mapping_with_cp: Optional[Mapping]) -> torch.Tensor:
    """Restore full token count after the last layer's reduce-scatter.

    With Helix CP + Attention DP, each decoder layer's reduce-scatter
    leaves each CP rank with only its chunk of tokens.  This function
    performs an AllGather across the CP group so that the LM head (and
    final norm) see every token.

    Should be called at the end of the model's ``forward()`` method,
    after the decoder layer loop.
    """
    if (mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
            and mapping_with_cp.enable_attention_dp):
        hidden_states = cp_allgather(hidden_states, mapping_with_cp, dim=0)
        hidden_states = hidden_states[:attn_metadata.num_tokens]
    return hidden_states


class Attention(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
        disable_deep_gemm: bool = False,
        attn_output_gate: Optional[bool] = None,
        use_custom_cublas_mm: bool = False,
        reduce_output: bool = True,
        mapping_with_cp: Optional[Mapping] = None,
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            pos_embd_params (Optional[PositionalEmbeddingParams]): The positional embedding parameters.
            rope_fusion (Optional[bool]): Whether to fuse RoPE into the attention OP and skip applying unfused RoPE. If None, whether to fuse is decided by the capability of the attention backend.
            layer_idx (Optional[int]): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (Optional[bool]): Whether to use bias in the output projection layer.
            config (Optional[ModelConfig]): The model configuration.
            q_scaling (float): The scaling factor for the qk_scale. The definition is $O = softmax(QK^T * qk_scale) * V, qk_scale = 1 / (sqrt(head_dim) * q_scaling)$. The default value is 1.0.
            attention_chunk_size (Optional[int]): See [Chunked Attention] below.
            disable_deep_gemm (bool): Whether to disable the use of DeepGEMM in Linear layers (currently only matters on SM100 + FP8).
            attn_output_gate (Optional[bool]): Determines whether to use an output gate in the attention Op. If False, the decision is automatically handled by the attention backend based on its capabilities.
            mapping_with_cp (Optional[Mapping]): Override mapping with CP configuration.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)

        self.register_to_config = False
        # We only register TRTLLM attention layers to config.
        if config is not None:
            if "attn_layers" not in config.extra_attrs:
                config.extra_attrs["attn_layers"] = {}
            suffix = 0
            # Makes sure there is no duplicate attention layer identifier.
            while self.layer_idx_str in config.extra_attrs["attn_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            config.extra_attrs["attn_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = getattr(config.pretrained_config, 'head_dim', None)
        if not isinstance(self.head_dim, int):
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.q_scaling = q_scaling
        self.attn_output_gate = attn_output_gate

        if self.attn_output_gate:
            logger.info_once("using attn output gate!", key="attn_output_gate")

        # [Chunked Attention]
        # Chunked attention is applied to context requests only. Chunked attention will be
        # applied when this field is specified and mMaskType == CAUSAL.
        #
        # In chunked attention, we break context requests into chunks of a specified size. Tokens can only
        # attend to tokens in the same chunk. So, for example, if the chunk size is 3, we might have a mask
        # that looks like this:
        #
        # 1 0 0 0 0 0
        # 1 1 0 0 0 0
        # 1 1 1 0 0 0
        # 0 0 0 1 0 0
        # 0 0 0 1 1 0
        # 0 0 0 1 1 1
        self.attention_chunk_size = attention_chunk_size

        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        if mapping_with_cp is not None:
            logger.warning_once(
                "[Attention::__init__] Overriding mapping with CP detected.",
                key="attention_init_mapping_with_cp")
            self.mapping = mapping_with_cp
        else:
            self.mapping = config.mapping

        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        cp_size = self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1

        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(
            ), f"CP type must be HELIX for Attention, but got {self.mapping.cp_config['cp_type']}."

        mapping = Mapping(
            world_size=dp_size * tp_size * pp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.tp_rank = mapping.tp_rank
        assert self.num_heads % (tp_size * cp_size) == 0
        self.num_heads = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads // cp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm

        qkv_shard_indices_mapping = {
            "q": (0, self.q_size * (2 if self.attn_output_gate else 1)),
            "k":
            (self.q_size * (2 if self.attn_output_gate else 1), self.kv_size),
            "v":
            (self.q_size * (2 if self.attn_output_gate else 1) + self.kv_size,
             self.kv_size),
        }

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size * (2 if self.attn_output_gate else 1) +
            2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm,
            fused_weight_shard_indices_mapping=qkv_shard_indices_mapping,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm)

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        # For Helix CP, combine TP and CP for the output projection so each
        # rank's o_proj input is num_heads_tp_cp * head_dim.
        mapping_o = Mapping(
            world_size=dp_size * tp_size * pp_size * cp_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = mapping_o

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping_o,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.o_lora,
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm)

        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend

        # Resolve target_sparsity → threshold_scale_factor if needed
        sparse_attn_cfg = config.sparse_attention_config
        if (isinstance(sparse_attn_cfg, SkipSoftmaxAttentionConfig)
                and sparse_attn_cfg.target_sparsity is not None):
            hf_sparse = getattr(config.pretrained_config,
                                'sparse_attention_config', None)
            if not isinstance(hf_sparse, dict):
                raise ValueError(
                    "sparse_attention_config with target_sparsity requires formula "
                    "coefficients in the model's config.json "
                    "(sparse_attention_config.threshold_scale_factor.{prefill,decode}.{a,b}), "
                    "but sparse_attention_config was not found or was not dict type in config.json."
                )
            formula = hf_sparse.get('threshold_scale_factor', {})
            sparse_attn_cfg = sparse_attn_cfg.resolve_for_target_sparsity(
                formula)

        attn_cls = get_attention_backend(self.attn_backend,
                                         sparse_attn_config=sparse_attn_cfg)

        # These two modules are mutually exclusive - either splitted_qkv_lora or fused_qkv_lora will be used,
        # but never both at the same time. splitted_qkv_lora handles Q,K,V separately while fused_qkv_lora
        # handles them as a single fused operation.
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.kv_size])
        self.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV],
                                        [self.q_size + 2 * self.kv_size])

        # Whether to fuse RoPE into the attention OP.
        # If true, RoPE will be applied in self.attn.forward.
        # If false, RoPE will be applied in self.apply_rope.
        self.rope_fusion = rope_fusion

        if config.sparse_attention_config is not None:
            # Log sparse attention configuration once
            algo = config.sparse_attention_config.algorithm
            cfg_dump = config.sparse_attention_config.model_dump(
                exclude_none=True)
            logger.info_once(f"Using sparse attention: {algo} {cfg_dump}",
                             key="sparse_attention_config")

            if config.sparse_attention_config.algorithm == "rocket":
                logger.warning_once("disable rope_fusion for RocketKV.",
                                    key="disable_rope_fusion_for_rocketkv")
                self.rope_fusion = False

        if self.rope_fusion and not attn_cls.support_fused_rope():
            logger.warning_once(
                "rope_fusion is true but the attention backend does not support it. Will disable rope_fusion.",
                key="disable_rope_fusion_for_non_supported_backend")
            self.rope_fusion = False
        # If rope_fusion is not specified, enable if the attention backend supports it.
        if self.rope_fusion is None:
            self.rope_fusion = attn_cls.support_fused_rope()

        self.rotary_emb = None
        if not self.rope_fusion and self.pos_embd_params is not None:
            if self.pos_embd_params.type.is_mrope():
                self.rotary_emb = MRotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                    mrope_section=self.pos_embd_params.mrope_section,
                    mrope_interleaved=self.pos_embd_params.mrope_interleaved)
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                )

        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params if self.rope_fusion else None,
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            q_scaling=self.q_scaling,
            attention_chunk_size=self.attention_chunk_size,
            sparse_attention_config=sparse_attn_cfg,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

        self.o_proj.create_weights()
        self.has_quant_scale = (self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4
                                or self.o_proj.has_fp8_block_scales
                                or self.o_proj.has_fp8_rowwise
                                or self.o_proj.has_w4a8_nvfp4_fp8)

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def _use_quantize_output(self):
        # If o_proj can't consume, then no need to quantize the output to nvfp4
        if hasattr(self.attn, 'has_nvfp4'
                   ) and self.attn.has_nvfp4 and not self.o_proj.has_nvfp4:
            return False

        # If o_proj does dynamic activation quantization, it computes its own scales
        # at runtime from a BF16 input — attention must NOT pre-quantize output
        if self.o_proj.force_dynamic_quantization:
            return False

        # If no quant is applied, no need to quantize the output
        if self.quant_config is not None and not self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            return False

        has_awq_pre_quant_scale = hasattr(
            self.o_proj,
            'pre_quant_scale') and self.o_proj.pre_quant_scale is not None

        return self.has_quant_scale and not self.attn_output_gate and not has_awq_pre_quant_scale

    def create_output(self, q: torch.Tensor, attn_metadata: AttentionMetadata,
                      mask_type: str):
        # Attention is treated as mixed request by default.
        return self.attn.create_output(
            q,
            is_quantize_output=self._use_quantize_output(),
            metadata=attn_metadata,
            attention_mask=mask_type,
            is_gen_only=False)

    def _helix_post_process(self, partial_o: torch.Tensor,
                            softmax_stats: torch.Tensor) -> torch.Tensor:
        """Helix CP post-processing: all-to-all exchange and combine partial
        attention outputs across CP ranks."""
        return _helix_post_process(partial_o, softmax_stats, self.mapping,
                                   self.num_heads_tp_cp, self.head_dim)

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        has_lora: bool = False,
    ):
        num_tokens = attn_metadata.num_tokens

        q = q[:num_tokens, :]
        if k is not None:
            k = k[:num_tokens, :]
        if v is not None:
            v = v[:num_tokens, :]

        mrope_config = None
        if mrope_rotary_cos_sin is not None or mrope_position_deltas is not None:
            mrope_config = dict()
            if mrope_rotary_cos_sin is not None:
                mrope_config["mrope_rotary_cos_sin"] = mrope_rotary_cos_sin
            if mrope_position_deltas is not None:
                mrope_config["mrope_position_deltas"] = mrope_position_deltas

        # Helix CP generation path: get partial outputs with softmax stats,
        # then exchange and combine across CP ranks.
        # NOTE: The helix post-process combine step works on unquantized
        # (BF16/FP16) partial outputs and softmax stats from each rank.
        # We intentionally skip passing out_scale/out_scale_sf to FMHA here
        # so it produces BF16 output. After combining, the downstream o_proj
        # linear layer handles quantization (FP8/NVFP4) in its apply() method.
        if self.mapping.has_cp_helix() and attn_metadata.num_contexts == 0:
            assert output is None, (
                "Helix produces BF16 partial outputs which may not match a pre-allocated FP8/NVFP4 buffer for torch.compile inplace output."
            )
            softmax_stats = torch.empty((num_tokens, self.num_heads, 2),
                                        device=q.device,
                                        dtype=torch.float32)
            attn_output = self.attn.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_mask=attention_mask,
                    mrope_config=mrope_config,
                    attention_window_size=attention_window_size,
                    attention_mask_data=attention_mask_data,
                    softmax_stats_tensor=softmax_stats,
                    attention_sinks=attention_sinks,
                ))
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = self._helix_post_process(attn_output, softmax_stats)
            return attn_output, None

        out_scale = None
        out_scale_sf = None
        # Don't set out_scale if o_proj has pre_quant_scale - this prevents FP8/FP4 output
        # and keeps attention output in BF16 for better precision when applying pre_quant_scale
        # Also don't set out_scale if LoRA is active - LoRA grouped_gemm doesn't support FP8
        if self._use_quantize_output() and not has_lora:
            out_scale = self.o_proj.inv_input_scale
            out_scale_sf = self.o_proj.input_scale

        kv_scales_sf = None
        kv_scales_sf_inv = None
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp4_kv_cache(
        ):
            kv_scales_sf = self.qkv_proj.kv_scales
            kv_scales_sf_inv = self.qkv_proj.inv_kv_scales

        attn_output = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                out_scale=out_scale,
                out_scale_sf=out_scale_sf,
                kv_scales_sf=kv_scales_sf,
                kv_scales_sf_inv=kv_scales_sf_inv,
                attention_mask=attention_mask,
                mrope_config=mrope_config,
                attention_window_size=attention_window_size,
                attention_mask_data=attention_mask_data,
                output=output[:num_tokens, :] if output is not None else None,
                output_sf=output_sf,
                attention_sinks=attention_sinks,
            ))
        if isinstance(attn_output, tuple):
            assert len(
                attn_output
            ) == 2, "attn_output should be a tuple of (output, output_sf)"
            return attn_output[0], attn_output[1]
        return attn_output, None

    def forward_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        mrope_config: Optional[dict],
        attention_sinks: Optional[torch.Tensor] = None,
        has_lora: bool = False,
    ):
        mrope_rotary_cos_sin = None
        mrope_position_deltas = None
        if mrope_config is not None:
            if "mrope_rotary_cos_sin" in mrope_config:
                mrope_rotary_cos_sin = mrope_config["mrope_rotary_cos_sin"]
            if "mrope_position_deltas" in mrope_config:
                mrope_position_deltas = mrope_config["mrope_position_deltas"]

        # Currently only TRTLLM and FLASHINFER are torch compile compatible backends.
        # Only enable custom inplace op when torch compiling.
        use_custom_inplace_op = (self.register_to_config
                                 and (self.attn_backend == "TRTLLM"
                                      or self.attn_backend == "FLASHINFER")
                                 and is_torch_compiling())

        if use_custom_inplace_op:
            outputs = create_attn_outputs(q, attention_mask, self.layer_idx_str)
            assert len(outputs) == 1 or len(outputs) == 2
            output = outputs[0]
            output_sf = outputs[1] if len(outputs) == 2 else None
            attn_custom_op_inplace(
                q,
                k,
                v,
                attention_mask,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                attention_window_size,
                attention_mask_data,
                attention_sinks,
                self.layer_idx_str,
                output,
                output_sf,
            )
        else:
            output, output_sf = self._attn_impl(q,
                                                k,
                                                v,
                                                attn_metadata,
                                                attention_mask,
                                                mrope_rotary_cos_sin,
                                                mrope_position_deltas,
                                                attention_window_size,
                                                attention_mask_data,
                                                attention_sinks=attention_sinks,
                                                has_lora=has_lora)
        if output_sf is not None:
            output = Fp4QuantizedTensor(output, output_sf)

        return output

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            attention_mask (AttentionMask): The attention mask type.
            mrope_config (Optional[dict]): The MROPE configuration.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            lora_params (Optional[dict]): The LoRA parameters.
            attention_window_size (Optional[int]): The attention window size.
            attention_mask_data (Optional[torch.Tensor]): The attention mask data.
        Returns:
            torch.Tensor: The output tensor.
        """
        hidden_states = _helix_cp_allgather_input(hidden_states, attn_metadata,
                                                  self.mapping, self.layer_idx)

        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                              self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            # Single line: view -> chunk -> reshape both q and gate
            q, gate = [
                t.reshape(*orig_shape, -1) for t in torch.chunk(
                    q_gate.view(*orig_shape, self.num_heads, -1), 2, dim=-1)
            ]
        else:
            q, k, v = qkv, None, None

        # For dynamic tree spec decoding with Python RoPE, adjust position_ids
        # to use tree offsets (same as C++ kernel: past_seq_len + offset).
        if (not self.rope_fusion
                and getattr(attn_metadata, 'is_spec_dec_dynamic_tree', False)
                and getattr(attn_metadata, 'use_spec_decoding', False)
                and getattr(attn_metadata, 'spec_decoding_position_offsets',
                            None) is not None
                and attn_metadata.spec_decoding_position_offsets.dim() ==
                1  # 1D layout ⇒ dynamic tree
                and position_ids is not None):
            position_ids = self._adjust_position_ids_for_spec_dec(
                position_ids, attn_metadata)

        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        if attention_sinks is not None:
            assert self.attn_backend == "TRTLLM", "Attention sinks are only supported for TRTLLM backend."

        attn_output = self.forward_impl(q,
                                        k,
                                        v,
                                        attn_metadata,
                                        attention_mask,
                                        attention_window_size,
                                        attention_mask_data,
                                        mrope_config=mrope_config,
                                        attention_sinks=attention_sinks,
                                        has_lora=bool(lora_params))

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        attn_output = _helix_cp_output_projection(self.o_proj, attn_output,
                                                  attn_metadata,
                                                  all_reduce_params,
                                                  self.mapping, self.mapping_o,
                                                  self.layer_idx, lora_params)
        return attn_output

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        Apply RoPE to the query and key.
        Depending on the implementation, q, k, v could be either fused (q, k, v = concat(q, k, v), None, None) or unfused (none of q, k, v is None).
        Before self.attn.forward, convert_qkv will be called to make sure that the format of (q, k, v) satisfies the requirement of self.attn.
        This method could be overridden in the subclass, in which extra functionalities such as q_norm/k_norm could be added.
        Args:
            q (torch.Tensor): The query tensor.
            k (Optional[torch.Tensor]): The key tensor.
            v (Optional[torch.Tensor]): The value tensor.
            position_ids (torch.Tensor): The position IDs of each token for RoPE.
        Returns:
            tuple: A tuple of (q, k, v).
        """
        # If RoPE is fused into the attention OP, do not apply RoPE here.
        if not self.rope_fusion and position_ids is not None:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.rotary_emb(position_ids, [q, k])
        return q, k, v

    def _adjust_position_ids_for_spec_dec(self, position_ids, attn_metadata):
        """Replicate C++ kernel's rotary_pos = past_seq_len + offset."""
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        if num_gens <= 0:
            return position_ids
        gen_len = int(attn_metadata.seq_lens[num_contexts])
        base_pos = attn_metadata.kv_lens_cuda[num_contexts:num_contexts +
                                              num_gens] - gen_len
        offsets = attn_metadata.spec_decoding_position_offsets[:num_gens *
                                                               gen_len].view(
                                                                   num_gens,
                                                                   gen_len)
        adjusted = (base_pos.unsqueeze(1) + offsets).reshape(-1)
        start = attn_metadata.num_ctx_tokens
        end = start + num_gens * gen_len
        position_ids[0, start:end] = adjusted
        return position_ids

    def apply_qk_norm(self, q, k):
        raise NotImplementedError(
            f"QK norm is not implemented for {self.__class__.__name__}. "
            "Please override the `apply_qk_norm` method in the subclass.")


@torch.library.custom_op("trtllm::mla_custom_op_inplace",
                         mutates_args=("output", ))
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
    latent_cache_gen: Optional[torch.Tensor],
) -> None:
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.forward_impl(position_ids,
                           hidden_states,
                           metadata,
                           output=output,
                           latent_cache_gen=latent_cache_gen)


@torch.library.custom_op("trtllm::mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and conditionally
    indexer.pre_indexer_proj (FP8/FP4 quantize, weight scaling).  Does NOT
    update the indexer k cache — that happens in Op 2 (mla_dsa_attn_inplace)
    because the scatter kernel accesses batch-specific metadata.

    Returns [q, compressed_kv, k_pe, latent_cache] when the short-MHA path
    handles all tokens, or [q, compressed_kv, k_pe, latent_cache, q_fp8,
    k_fp8, k_scale, weights, q_scale] when the indexer runs.  Under torch
    compile, _should_use_short_mha returns False so the result is always
    length 9, keeping control flow straight-line for CUDA graph capture.
    The trailing q_scale is only consumed by the FP4 dispatch; the FP8
    path ignores it in forward_dsa_attn.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    return mla_layer.forward_dsa_proj(position_ids, hidden_states, metadata)


@mla_dsa_proj.register_fake
def _mla_dsa_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    # Under torch compile _should_use_short_mha is False, so the result is
    # always 9 tensors (4 attention inputs + 5 indexer intermediates, with
    # q_scale as the 9th carried for the FP4 dispatch).
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    num_tokens = hidden_states.shape[0]
    indexer = mla_layer.mqa.indexer
    q = hidden_states.new_empty(
        [num_tokens, mla_layer.num_heads_tp * mla_layer.qk_head_dim])
    compressed_kv = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank])
    k_pe = hidden_states.new_empty([num_tokens, mla_layer.qk_rope_head_dim])
    latent_cache = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank + mla_layer.qk_rope_head_dim])
    # Indexer intermediates: q_fp8, k_fp8, k_scale, weights, q_scale.
    # Under FP4 q_fp8's trailing dim is head_dim // 2 (two E2M1 codes per
    # byte) and q_scale carries one int32 per (token, head) packing four
    # UE8M0 exponents; under FP8 q_fp8's trailing dim is head_dim and
    # q_scale carries one float32 per (token, head).
    if indexer.use_fp4:
        q_fp8 = hidden_states.new_empty(
            [num_tokens, indexer.n_heads, indexer.head_dim // 2],
            dtype=torch.int8)
        k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim // 2],
                                        dtype=torch.int8)
        k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.int32)
        q_scale = hidden_states.new_empty([num_tokens, indexer.n_heads, 1],
                                          dtype=torch.int32)
    else:
        q_fp8 = hidden_states.new_empty(
            [num_tokens, indexer.n_heads, indexer.head_dim],
            dtype=torch.float8_e4m3fn)
        k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim],
                                        dtype=torch.float8_e4m3fn)
        k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.float32)
        q_scale = hidden_states.new_empty([num_tokens, indexer.n_heads, 1],
                                          dtype=torch.float32)
    weights = hidden_states.new_empty([num_tokens, indexer.n_heads],
                                      dtype=torch.float32)
    return [
        q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights,
        q_scale
    ]


@torch.library.custom_op("trtllm::mla_dsa_attn_inplace",
                         mutates_args=("output", ))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale] when
    the indexer ran in Op 1, or [] when short-MHA handled all tokens. The
    trailing q_scale is only consumed by the FP4 dispatch; the FP8 path
    ignores it. Runs sparse_attn_indexer then dispatches context/generation
    attention. This op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.forward_dsa_attn(q, compressed_kv, k_pe, latent_cache,
                               indexer_intermediates, position_ids, metadata,
                               output)


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
    mat2_dequant: Optional[torch.Tensor] = None,
    use_cute_dsl_blockscaling_bmm: bool = False,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90 or sm_version == 89:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            mat1)

        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale,
                                                   output)
        out.copy_(output)
    elif sm_version == 120:
        mat1_fp8, mat1_scale = fp8_utils.per_token_quant_and_transform(
            mat1, need_permute102=True)
        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale,
                                                   output)
        out.copy_(output)
    elif is_sm_100f(sm_version):
        if use_cute_dsl_blockscaling_bmm:
            mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                mat1)
            torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(mat1_fp8, mat2_fp8,
                                                        mat1_scale, mat2_scale,
                                                        out)
            mat1_scale = None
        else:
            torch.bmm(mat1.transpose(0, 1),
                      mat2_dequant.transpose(1, 2),
                      out=out)
    else:
        raise NotImplementedError(f"SM{sm_version} is not supported")


class MLA(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream: Optional[torch.cuda.Stream] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ):
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream for running operations in two parallel streams.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        # Currently only DSA sparse attention is supported.
        if config is not None and config.sparse_attention_config is not None and config.sparse_attention_config.algorithm == "dsa":
            self.is_dsa = True
        else:
            self.is_dsa = False

        # tensor parallel
        config = config or ModelConfig()
        if mapping_with_cp is not None:
            logger.warning_once(
                "[MLA::__init__] Overriding mapping with CP detected.",
                key="mla_init_mapping_with_cp")
            self.mapping = mapping_with_cp
        else:
            self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        cp_size = self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1
        if self.mapping.has_cp_ulysses():
            raise NotImplementedError("MLA doesn't support CP Ulysses yet")
        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(
            ), f"CP type must be HELIX for MLA, but got {self.mapping.cp_config['cp_type']}."

        mapping = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        assert self.num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        self.num_key_value_heads_tp = (self.num_key_value_heads + tp_size -
                                       1) // tp_size

        rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", 1e-6)
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm

        if not self.is_lite:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm)

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)

            self.q_b_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm)
        else:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm)

            self.q_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm)
            self.q_b_proj = self.q_proj

        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm)
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        # Used in forward_absorption only
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )

        mapping_o = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = mapping_o
        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping_o,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            sparse_attention_config=config.sparse_attention_config,
            dtype=dtype,
            aux_stream=aux_stream,
        )

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.rope_fusion = self.mqa.support_fused_rope()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        # Short-sequence MHA optimization for DSA models:
        # For short prefill sequences, use MHA (kv_b_proj expansion + standard
        # attention) instead of the absorption path, which has overhead from
        # extra BMMs and larger head_dim (kv_lora_rank + qk_rope_head_dim).
        # Only active when rope_fusion is True (DSA with TrtllmAttention).
        _threshold_str = os.environ.get('TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD',
                                        '0')
        try:
            self.short_seq_mha_threshold = int(_threshold_str)
        except ValueError as err:
            raise ValueError(
                f"TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD must be an integer, "
                f"got '{_threshold_str}'") from err

        # MHA attention backend: used by non-DSA (standard MLA) and optionally
        # by DSA for the short-seq path (dense attention, no sparse config).
        _short_seq_mha = (self.is_dsa and self.short_seq_mha_threshold > 0
                          and not self.apply_rotary_emb)
        if not self.is_dsa or _short_seq_mha:
            self.mha = create_attention(
                config.attn_backend,
                self.layer_idx,
                self.num_heads_tp,
                head_dim=self.qk_head_dim,
                num_kv_heads=self.num_key_value_heads_tp,
                pos_embd_params=pos_embd_params,
                quant_config=quant_config,
                q_scaling=q_scaling,
                is_mla_enable=True,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                predicted_tokens_per_seq=self.predicted_tokens_per_seq,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                sparse_attention_config=(None if _short_seq_mha else
                                         config.sparse_attention_config),
            )
        else:
            self.mha = None

        self.llama_4_scaling = False
        if hasattr(config.pretrained_config, 'llama_4_scaling'):
            self.llama_4_scaling = True
            self.floor_scale = getattr(config.pretrained_config.llama_4_scaling,
                                       'original_max_position_embeddings', 8192)
            self.attn_scale = getattr(config.pretrained_config.llama_4_scaling,
                                      'beta', 0.1)

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        # self.mha is non-None for non-DSA models (standard MHA) and for DSA
        # models when the short-seq MHA optimization is active.
        if self.mha is not None:
            self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())

        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp_cp,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp, self.kv_lora_rank,
                         self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp_cp, self.v_head_dim,
                         self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads_tp, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:].reshape(
            -1, self.num_heads_tp * self.qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim:] = q_pe.view(-1, self.num_heads_tp,
                                                   self.qk_rope_head_dim)
        return k_pe

    def _attn_forward_gen(self, attn_backend: AttentionBackend, q: torch.Tensor,
                          k: torch.Tensor, v: torch.Tensor,
                          position_ids: Optional[torch.Tensor],
                          attn_metadata: AttentionMetadata, **kwargs):
        if self.mapping.has_cp_helix():
            # partial_o: [num_tokens, num_heads_tp * kv_lora_rank]
            # softmax_stats: [num_tokens, num_heads_tp, 2]
            softmax_stats = torch.empty((q.shape[0], self.num_heads_tp, 2),
                                        device=q.device,
                                        dtype=torch.float32)
            kwargs["softmax_stats_tensor"] = softmax_stats
            partial_o = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            kv_lora_rank = partial_o.shape[-1] // self.num_heads_tp
            assert self.kv_lora_rank == kv_lora_rank

            return _helix_post_process(partial_o, softmax_stats, self.mapping,
                                       self.num_heads_tp_cp, kv_lora_rank,
                                       self.aux_stream, self.ln_events)
        else:
            attn_output = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            return attn_output

    def create_output(self, hidden_states: torch.Tensor, num_contexts: int):
        num_tokens = hidden_states.shape[0]
        hidden_size = self.o_proj.in_features
        return hidden_states.new_empty([num_tokens, hidden_size],
                                       dtype=hidden_states.dtype)

    def _attention_scaling(self, q, position_ids):

        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def forward_impl(self,
                     position_ids: Optional[torch.Tensor],
                     hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     output: torch.Tensor,
                     latent_cache_gen: Optional[torch.Tensor] = None) -> None:
        """
        Forward pass for the MLA module. Writes result into output tensor in-place.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            output (torch.Tensor): The output tensor to write results into.
            latent_cache_gen (Optional[torch.Tensor]): The latent cache used in generation.
        """
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = hidden_states[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(
                hidden_states).split([
                    self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
                ], -1)

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self.q_a_layernorm(q),
                lambda: self.kv_a_layernorm(compressed_kv),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        assert q.shape[
            0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            if self.llama_4_scaling:
                q_ctx = self._attention_scaling(
                    q_ctx, position_ids[..., :num_ctx_tokens])

            self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                position_ids,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            if latent_cache_gen is None:
                latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            if self.llama_4_scaling:
                q_gen = self._attention_scaling(
                    q_gen, position_ids[..., num_ctx_tokens:])

            self.forward_absorption_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
            )

    def forward_impl_with_dsa(self, position_ids: Optional[torch.Tensor],
                              hidden_states: torch.Tensor,
                              attn_metadata: AttentionMetadata,
                              output: torch.Tensor) -> None:
        """
        Forward pass for the MLA module with DSA (always in MQA mode).
        Writes result into output tensor in-place.

        Delegates to forward_dsa_proj (token-wise projections) followed by
        forward_dsa_attn (batch-dependent attention dispatch).

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            output (torch.Tensor): The output tensor to write results into.
        """
        proj_outputs = self.forward_dsa_proj(position_ids, hidden_states,
                                             attn_metadata)
        q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
        indexer_intermediates = proj_outputs[4:]
        self.forward_dsa_attn(q, compressed_kv, k_pe, latent_cache,
                              indexer_intermediates, position_ids,
                              attn_metadata, output)

    def forward_dsa_proj(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> List[torch.Tensor]:
        """Token-wise projections for DSA MLA (CUDA-graph-capturable Op 1).

        Runs kv_a_proj, layernorms, q_b_proj, and conditionally
        indexer.pre_indexer_proj().

        IMPORTANT: This method must NOT slice tensors by num_tokens or
        access batch-specific metadata, so that all operations are
        unconditionally straight-line for CUDA graph capture.  Slicing
        to num_tokens happens in forward_dsa_attn (Op 2, outside graph).

        Returns [q, compressed_kv, k_pe, latent_cache] when short-MHA
        handles all tokens (eager only), or
        [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
        weights] when the indexer runs.  Under torch compile
        _should_use_short_mha returns False so it is always length 8.
        """
        assert self.mqa is not None, "DSA is only supported in MQA mode"

        q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1)

        q, compressed_kv = maybe_execute_in_parallel(
            lambda: self.q_a_layernorm(q),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        qr = q
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

        q = self.q_b_proj(q)

        use_short_mha_for_ctx = self._should_use_short_mha(
            attn_metadata, position_ids)

        # Skip the indexer when the short MHA path handles all context
        # tokens and there are no generation tokens.
        if use_short_mha_for_ctx and attn_metadata.num_generations == 0:
            return [q, compressed_kv, k_pe, latent_cache]

        # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
        # compute (cublas_mm, rope, FP4/FP8 quantize, weight scaling) with no
        # access to batch-specific metadata or the k cache. Returns q_scale
        # as a 5th element so the FP4 dispatch can forward it to the kernel;
        # the FP8 path ignores it in forward_dsa_attn.
        q_fp8, k_fp8, k_scale, weights, q_scale = (
            self.mqa.indexer.pre_indexer_proj(qr, hidden_states, position_ids))

        return [
            q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
            weights, q_scale
        ]

    def forward_dsa_attn(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        indexer_intermediates: List[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Batch-structure-dependent attention for DSA MLA (Op 2, not graph-captured).

        indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale]
        when the indexer ran in Op 1, or [] when short-MHA handled all tokens.

        All num_tokens slicing happens here (not in Op 1) because
        num_tokens comes from batch-specific metadata and must not be
        baked into CUDA graph capture.
        """
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        # Slice Op 1 outputs to actual num_tokens (Op 1 operates on the
        # full padded tensor for CUDA graph compatibility).
        q = q[:num_tokens, ...]
        compressed_kv = compressed_kv[:num_tokens, ...]
        k_pe = k_pe[:num_tokens, ...]
        latent_cache = latent_cache[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        use_short_mha_for_ctx = (num_contexts > 0
                                 and self._should_use_short_mha(
                                     attn_metadata, position_ids))

        if use_short_mha_for_ctx and num_generations == 0:
            topk_indices = None
        else:
            q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
            # Slice indexer intermediates to actual num_tokens (they were
            # computed on the full padded tensor in Op 1).
            q_fp8 = q_fp8[:num_tokens, ...]
            k_fp8 = k_fp8[:num_tokens, ...]
            k_scale = k_scale[:num_tokens, ...]
            weights = weights[:num_tokens, ...]
            q_scale = q_scale[:num_tokens, ...]
            topk_indices = self.mqa.indexer.sparse_attn_indexer(
                attn_metadata,
                q,  # only used for shape/device in buffer allocation
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                q_scale=q_scale,
            )

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            self.forward_context_dsa(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
                topk_indices=topk_indices[:num_ctx_tokens, :]
                if topk_indices is not None else None,
                position_ids=position_ids,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            self.forward_generation_dsa(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                latent_cache_gen,
                topk_indices=topk_indices[num_ctx_tokens:num_tokens, :],
            )

    def forward_context_default(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dense MHA context path: expand KV via kv_b_proj and run attention.

        Used by non-DSA models and as the short-seq MHA fallback for DSA models.
        """
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim
            ],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads_tp, self.qk_head_dim)
        maybe_compiled_copy_(
            k[..., :self.qk_nope_head_dim],
            k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim))
        # When rope_fusion=True (apply_rotary_emb=False), the rope portion
        # of k is left uninitialized here; the fused attention kernel
        # handles k_pe RoPE via latent_cache instead.
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                       self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def _should_use_short_mha(self, attn_metadata: AttentionMetadata,
                              position_ids: Optional[torch.Tensor]) -> bool:
        """Check if the short-seq MHA optimization should be used for context.

        Uses max_ctx_kv_len (max total KV length per context sequence,
        including cached tokens) when available, to correctly account for
        chunked context where the full attention span exceeds the threshold
        even if the new token count is small.  Falls back to num_ctx_tokens
        (total new context tokens) when max_ctx_kv_len is not set.

        Disabled under torch compile so that the split DSA custom ops
        (mla_dsa_proj / mla_dsa_attn_inplace) have unconditionally
        straight-line control flow for CUDA graph capture.
        """
        if is_torch_compiling():
            return False
        if not (self.short_seq_mha_threshold > 0 and not self.apply_rotary_emb
                and self.mapping.cp_size == 1 and position_ids is not None):
            return False
        effective_len = getattr(attn_metadata, 'max_ctx_kv_len',
                                attn_metadata.num_ctx_tokens)
        return effective_len <= self.short_seq_mha_threshold

    def forward_context_dsa(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run context-phase attention for DSA models.

        Dispatches to the short-seq MHA path (forward_context) when the max
        per-sequence KV length (including cached tokens) is within the
        threshold, or falls through to the absorption/sparse MLA path
        otherwise.  forward_context() further dispatches to the appropriate
        handler (forward_context_default, forward_context_with_cached_kv, or
        forward_context_with_chunked_prefill) based on cached-KV state.

        Args:
            q: Query tensor, shape [num_ctx_tokens, num_heads * qk_head_dim].
            compressed_kv: Latent KV, shape [num_ctx_tokens, kv_lora_rank].
            k_pe: RoPE key portion, shape [num_ctx_tokens, qk_rope_head_dim].
            attn_metadata: Attention metadata for the current batch.
            output: Pre-allocated output tensor, written in-place.
            latent_cache: Concatenated [compressed_kv, k_pe] for KV cache.
            topk_indices: Sparse routing indices from the indexer (None when
                the short-seq MHA path is used).
            position_ids: Token position IDs (required for short-seq MHA).
        """
        # Short-sequence MHA: bypass absorption path for short prefills,
        # using kv_b_proj expansion + standard attention instead.
        # See __init__ comment for rationale. topk_indices is not used
        # because dense attention is faster than sparse routing at this scale.
        # forward_context() handles cached tokens by dispatching to
        # forward_context_with_cached_kv or forward_context_with_chunked_prefill.
        if self._should_use_short_mha(attn_metadata, position_ids):
            return self.forward_context(q, compressed_kv, k_pe, position_ids,
                                        attn_metadata, output, latent_cache)

        if get_sm_version() >= 100:
            return self.forward_absorption_context(q,
                                                   compressed_kv,
                                                   k_pe,
                                                   attn_metadata,
                                                   output,
                                                   latent_cache=latent_cache,
                                                   topk_indices=topk_indices)
        else:
            return self.forward_sparse_mla_kvcache_bf16(q,
                                                        latent_cache,
                                                        attn_metadata,
                                                        output,
                                                        topk_indices,
                                                        is_generation=False)

    def forward_generation_dsa(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_sm_version() >= 100:
            return self.forward_absorption_generation(q,
                                                      compressed_kv,
                                                      k_pe,
                                                      attn_metadata,
                                                      output,
                                                      latent_cache=latent_cache,
                                                      topk_indices=topk_indices)
        else:
            return self.forward_sparse_mla_kvcache_bf16(q,
                                                        latent_cache,
                                                        attn_metadata,
                                                        output,
                                                        topk_indices,
                                                        is_generation=True)

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype)
        assert full_compressed_kv.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert full_k_pe.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim
            ],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_heads_tp,
                                       self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = maybe_compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1)
        full_k = full_k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        attn_output = self.mha.forward(
            q,
            full_k,
            full_v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.
        Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [total_token_q, num_heads, 2] -> [total_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device='cuda',
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device='cuda',
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads_tp * self.v_head_dim), dtype=q.dtype)

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0]

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts]
            chunked_global_offset = attn_metadata.chunked_global_offset[
                loop_idx]
            chunked_max_seq_len = attn_metadata.max_chunk_len_per_loop[loop_idx]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype)

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [
                    self.num_heads_tp * self.qk_nope_head_dim,
                    self.num_heads_tp * self.v_head_dim
                ],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_heads_tp,
                                                 self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = maybe_compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads_tp,
                                                     -1)),
                dim=-1)
            chunked_k = chunked_k.view(-1, self.num_heads_tp * self.qk_head_dim)

            # release pytorch activation memory
            chunked_compressed_kv = None
            chunked_k_pe = None
            chunked_kv = None
            chunked_k_nope = None

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[
                loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[
                loop_idx]
            attn_metadata.host_total_kv_lens[0] = total_ctx_chunked_tokens

            # do not apply mask for attention within loop
            # latent_cache must be None to differentiate from normal context phase,
            # so that we can skip applying RoPE and appending KV cache inside attention op
            temp_attn_output = self.mha.forward(
                q,
                chunked_k,
                chunked_v,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    out_scale=self.out_scale,
                    attention_mask=PredefinedAttentionMask.FULL,
                    softmax_stats_tensor=self.temp_softmax_stats_tensor,
                    chunked_prefill_buffer_batch_size=attn_metadata.
                    runtime_features.chunked_prefill_buffer_batch_size,
                    output=temp_attn_output,
                ),
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output, temp_attn_output, self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor, temp_merge_op, attn_metadata)

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([
            -1, self.kv_lora_rank + self.qk_rope_head_dim
        ]).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        # final round of attention

        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim
            ],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = maybe_compiled_cat((k_nope, k_pe.expand(-1, self.num_heads_tp, -1)),
                               dim=-1)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[
            0] = attn_metadata.prompt_lens_cpu_runtime[:attn_metadata.
                                                       num_contexts].sum().item(
                                                       )

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        temp_attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                chunked_prefill_buffer_batch_size=attn_metadata.
                runtime_features.chunked_prefill_buffer_batch_size,
                output=temp_attn_output,
            ),
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(attn_output, temp_attn_output,
                                                 self.softmax_stats_tensor,
                                                 self.temp_softmax_stats_tensor,
                                                 temp_merge_op, attn_metadata)
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = origin_ctx_total_kv_len

        return attn_output

    @staticmethod
    @functools.cache
    def cached_warmup_forward_context_with_chunked_prefill(
            num_heads_tp, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank,
            v_head_dim, dtype, device):
        """Warmup torch.compile for cat operations with different tensor layouts.

        Tensors are marked with torch._dynamo.maybe_mark_dynamic(..., 0) on the
        num_tokens dimension, so for num_tokens != 1 a single warmup run is
        enough and the compiled kernel generalizes across varying num_tokens at
        runtime. num_tokens=1 still triggers recompile (torch.compile specializes
        for it), so it is warmed up separately. Do not use torch.compile with
        dynamic=True here because it completely ignores tensor layout/stride
        information, resulting in significantly degraded performance.
        """

        def warmup(num_tokens):
            chunked_k_nope = k_nope = torch.empty(
                num_tokens,
                num_heads_tp * (qk_nope_head_dim + v_head_dim),
                dtype=dtype,
                device=device)[:, :num_heads_tp * qk_nope_head_dim].view(
                    num_tokens, num_heads_tp, qk_nope_head_dim)
            chunked_k_pe = torch.empty(num_tokens,
                                       1,
                                       qk_rope_head_dim,
                                       dtype=dtype,
                                       device=device).expand(
                                           -1, num_heads_tp, -1)
            k_pe = torch.empty(num_tokens,
                               1,
                               kv_lora_rank + qk_rope_head_dim,
                               dtype=dtype,
                               device=device)[:, :, -qk_rope_head_dim:].expand(
                                   -1, num_heads_tp, -1)
            torch._dynamo.maybe_mark_dynamic(chunked_k_nope, 0)
            torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
            torch._dynamo.maybe_mark_dynamic(k_pe, 0)
            maybe_compiled_cat((chunked_k_nope, chunked_k_pe), dim=-1)
            maybe_compiled_cat((k_nope, k_pe), dim=-1)

        # With dim 0 (num_tokens) marked dynamic, one warmup suffices for all
        # num_tokens != 1 at runtime.
        warmup(2)

        # num_tokens=1 still triggers recompile; warm it separately.
        warmup(1)

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            if trtllm_attention.is_chunked_prefill_mla_context_for_warmup(
                    attn_metadata):
                self.cached_warmup_forward_context_with_chunked_prefill(
                    self.num_heads_tp, self.qk_nope_head_dim,
                    self.qk_rope_head_dim, self.kv_lora_rank, self.v_head_dim,
                    q.dtype, q.device)
            if trtllm_attention.is_chunked_prefill_for_mla_context(
                    attn_metadata) and get_sm_version() >= 100:
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output)
            elif trtllm_attention.has_cached_kv_for_mla_context(
                    attn_metadata
            ) or trtllm_attention.is_chunked_prefill_for_mla_context(
                    attn_metadata):
                return self.forward_context_with_cached_kv(
                    q, latent_cache, attn_metadata, output)
        return self.forward_context_default(q, compressed_kv, k_pe,
                                            position_ids, attn_metadata, output,
                                            latent_cache)

    def _bmm_bf16_out(self, a, b_no_transpose, b_transposed, output):
        """BMM with optional CuTe DSL bf16 acceleration on Blackwell."""
        if self.use_cute_dsl_bf16_bmm and is_sm_100f():
            torch.ops.trtllm.cute_dsl_bf16_bmm_blackwell(
                a, b_no_transpose, output)
        else:
            torch.ops.trtllm.bmm_out(a, b_transposed, output)

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        num_seqs = attn_metadata.kv_lens_cuda_runtime.size(0)

        cu_q_seqlens = torch.empty(num_seqs + 1,
                                   dtype=torch.int32,
                                   device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1,
                                    dtype=torch.int32,
                                    device=q.device)
        fmha_scheduler_counter = torch.empty(1,
                                             dtype=torch.uint32,
                                             device=q.device)
        has_fp8_kv_cache = self.mqa.has_fp8_kv_cache if hasattr(
            self.mqa, 'has_fp8_kv_cache') else False

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2,
                                         dtype=torch.float32,
                                         device=q.device)
            mla_bmm2_scale = torch.empty(1,
                                         dtype=torch.float32,
                                         device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self.num_heads_tp, (self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.uint8,
                device=q.device)

        fused_q = torch.empty(
            [
                num_tokens, self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        rope_stream = self.aux_stream if not has_fp8_kv_cache else None
        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            maybe_execute_in_parallel(
                lambda: self._bmm_bf16_out(q_nope_t, self.k_b_proj_trans,
                                           self.k_b_proj_trans.transpose(1, 2),
                                           q_nope_out),
                lambda: self.mqa.mla_rope_generation(
                    fused_q,
                    q_pe,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                ),
                self.ln_events[0],
                self.ln_events[1],
                rope_stream,
            )

        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            maybe_execute_in_parallel(
                lambda: fp8_block_scaling_bmm_out(
                    q_nope,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans_scale,
                    q_nope_out,
                    self.k_b_proj_trans_dequant,
                    self.use_cute_dsl_blockscaling_bmm,
                ),
                lambda: self.mqa.mla_rope_generation(
                    fused_q,
                    q_pe,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                ),
                self.ln_events[0],
                self.ln_events[1],
                rope_stream,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        fused_q = fused_q.view([
            num_tokens,
            self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.generation_only

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=attention_input_type,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            topk_indices=topk_indices,  # used by DSA attention
            is_generation=True,  # used by DSA attention
            cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
            cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
            fmha_scheduler_counter=
            fmha_scheduler_counter,  # used by `mlaGeneration`
            mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
            mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
            quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (attn_out_latent.shape[0] == q.shape[0]
                and attn_out_latent.shape[1]
                == self.num_heads_tp_cp * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view(
            [num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(attn_out_latent.transpose(0, 1), self.v_b_proj,
                               self.v_b_proj.transpose(1, 2),
                               attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward_absorption_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            self._bmm_bf16_out(q_nope_t, self.k_b_proj_trans,
                               self.k_b_proj_trans.transpose(1, 2), q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        if self.apply_rotary_emb:
            fused_q[..., self.kv_lora_rank:] = q_pe
        fused_q = fused_q.view([
            num_tokens,
            self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.context_only
        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=attention_input_type,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            topk_indices=topk_indices,  # used by DSA attention
            is_generation=False,  # used by DSA attention
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (attn_out_latent.shape[0] == q.shape[0]
                and attn_out_latent.shape[1]
                == self.num_heads_tp_cp * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view(
            [num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(attn_out_latent.transpose(0, 1), self.v_b_proj,
                               self.v_b_proj.transpose(1, 2),
                               attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    @nvtx_range("forward_sparse_mla_kvcache_bf16")
    def forward_sparse_mla_kvcache_bf16(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: torch.Tensor,
        is_generation: bool = False,
    ) -> torch.Tensor:
        """
        Forward sparse MLA (DSA) for BF16 KV cache for both context and generation phases using FlashMLA kernels

        To form the input for FlashMLA kernel and adapt our KV cache manager, we need to:
        1. Append current tokens to paged cache and apply rope to q/k via mla_rope_append_paged_kv_assign_q
        2. Load full kv cache from paged memory (with k rope applied)
        3. Call FlashMLA sparse attention kernel for sparse prefill/decode
        """
        assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), \
            "DSA requires DSAtrtllmAttentionMetadata"
        # Append current tokens to paged cache and apply RoPE to q
        # This writes latent_cache to paged KV and modifies q in-place
        trtllm_attention = self.mqa
        with nvtx_range_debug(
                f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"
        ):
            trtllm_attention.mla_rope_append_paged_kv_assign_q(
                q, latent_cache, attn_metadata, is_generation=is_generation)

        num_tokens = q.shape[0]
        q_nope, q_rope = q.view(-1, self.num_heads_tp, self.qk_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope_out = torch.empty(
            [num_tokens, self.num_heads_tp, (self.kv_lora_rank)],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            self._bmm_bf16_out(q_nope_t, self.k_b_proj_trans,
                               self.k_b_proj_trans.transpose(1, 2), q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        q_nope_out = q_nope_out.transpose(0, 1)
        q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

        sm_version = get_sm_version()
        # FlashMLA sparse kernel (bf16) requires num_heads=128 on sm100 or multiple of 64 on sm90
        if sm_version >= 100:
            padding = 128
            assert self.num_heads_tp <= padding, (
                f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
                f"got {self.num_heads_tp}. Padding from values > {padding} is not supported."
            )
        else:  # SM90
            padding = ((self.num_heads_tp + 63) // 64) * 64  # multiple of 64

        if self.num_heads_tp != padding:
            logger.warning_once(
                f"Padding num_heads from {self.num_heads_tp} to {padding} "
                f"due to FlashMLA sparse attention kernel requirement",
                key="sparse_mla_padding_warning")

            # Create padded tensor with zeros for extra heads
            q_padded = q_concat.new_empty(
                (num_tokens, padding, q_concat.shape[2]))
            q_padded[:, :self.num_heads_tp, :] = q_concat
            q_concat = q_padded

        # Convert indices and return all-layer KV pool
        # Note: underlying pool is layer-interleaved: [num_blocks, num_layers, kv_factor, tokens_per_block, num_kv_heads, head_dim]
        # to avoid reshape(copy) per-layer KV cache, we return all-layer KV pool w/ topk indices adjusted by stride_factor=num_layers*tokens_per_block
        topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            attn_metadata,
            layer_idx=self.layer_idx,
            is_generation=is_generation,
        )
        topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
        if flash_mla_sparse_fwd is not None:
            attn_out_latent = flash_mla_sparse_fwd(q_concat, kv_cache_pool,
                                                   topk_indices_pool,
                                                   self.softmax_scale)[0]
        else:
            raise RuntimeError(
                "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
            )

        # [seq, num_heads, kv_lora_rank], account for padding
        attn_out_latent = attn_out_latent[:, :self.num_heads_tp, :]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads_tp, self.kv_lora_rank])
        if self.num_heads_tp != padding:
            attn_out_latent = attn_out_latent.contiguous()

        assert (attn_out_latent.shape[0] == q.shape[0]
                and attn_out_latent.shape[1] == self.num_heads_tp)

        attn_output = output.view(
            [num_tokens, self.num_heads_tp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(attn_out_latent.transpose(0, 1), self.v_b_proj,
                               self.v_b_proj.transpose(1, 2),
                               attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")
        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hidden_states = _helix_cp_allgather_input(hidden_states, attn_metadata,
                                                  self.mapping, self.layer_idx)

        attn_output = self.create_output(hidden_states,
                                         attn_metadata.num_contexts)
        if self.register_to_config:
            if self.is_dsa:
                proj_outputs = torch.ops.trtllm.mla_dsa_proj(
                    hidden_states, position_ids, self.layer_idx_str)
                q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
                indexer_intermediates = proj_outputs[4:]
                torch.ops.trtllm.mla_dsa_attn_inplace(
                    q, compressed_kv, k_pe, latent_cache, indexer_intermediates,
                    position_ids, self.layer_idx_str, attn_output)
            else:
                torch.ops.trtllm.mla_custom_op_inplace(hidden_states,
                                                       position_ids,
                                                       self.layer_idx_str,
                                                       attn_output,
                                                       latent_cache_gen)
        elif self.is_dsa:
            self.forward_impl_with_dsa(position_ids,
                                       hidden_states,
                                       attn_metadata,
                                       output=attn_output)
        else:
            self.forward_impl(position_ids,
                              hidden_states,
                              attn_metadata,
                              output=attn_output,
                              latent_cache_gen=latent_cache_gen)

        attn_output = _helix_cp_output_projection(self.o_proj, attn_output,
                                                  attn_metadata,
                                                  all_reduce_params,
                                                  self.mapping, self.mapping_o,
                                                  self.layer_idx)
        return attn_output

    def resmooth_parameters(self,
                            module_weight,
                            module_weight_scale,
                            recipe=(1, 128, 128)):
        weight, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(
            module_weight, module_weight_scale)

        transfromed_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight.shape[1],
            k=weight.shape[2],
            recipe=recipe,
            num_groups=weight.shape[0],
            is_sfa=False)

        weight_param = torch.nn.Parameter(weight, requires_grad=False)
        scale_param = torch.nn.Parameter(transfromed_scale, requires_grad=False)

        return weight_param, scale_param

    def post_load_weights(self):
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())
        is_sm120 = get_sm_version() == 120
        if is_sm120 and has_fp8_block_scales:
            self.k_b_proj_trans, self.k_b_proj_trans_scale = self.resmooth_parameters(
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                recipe=(1, 128, 128))

            self.v_b_proj, self.v_b_proj_scale = self.resmooth_parameters(
                self.v_b_proj, self.v_b_proj_scale, recipe=(1, 128, 128))
''',
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
