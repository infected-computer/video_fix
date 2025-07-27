/*
 * PhoenixDRS Professional - High-Performance Pattern Matcher Implementation
 * מימוש מודול חיפוש דפוסים בביצועים גבוהים
 */

#include "FileCarver.h"
#include "ForensicLogger.h"
#include <QDebug>
#include <algorithm>
#include <queue>
#include <cstring>

// SIMD intrinsics
#include <emmintrin.h> // SSE2
#ifdef __AVX2__
#include <immintrin.h> // AVX2
#endif

namespace PhoenixDRS {

/*
 * PatternMatcher Constructor
 */
PatternMatcher::PatternMatcher()
    : m_root(std::make_unique<ACNode>())
    , m_automatonBuilt(false)
    , m_useSSE2(detectSSE2Support())
    , m_useAVX2(detectAVX2Support())
{
    PERF_LOG("PatternMatcher initialized - SSE2: %s, AVX2: %s", 
             m_useSSE2 ? "Yes" : "No", m_useAVX2 ? "Yes" : "No");
}

/*
 * PatternMatcher Destructor
 */
PatternMatcher::~PatternMatcher() = default;

/*
 * Detect SSE2 Support
 */
bool PatternMatcher::detectSSE2Support()
{
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[3] & (1 << 26)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (edx & (1 << 26)) != 0;
    }
    return false;
#endif
}

/*
 * Detect AVX2 Support
 */
bool PatternMatcher::detectAVX2Support()
{
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    return (cpuInfo[1] & (1 << 5)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & (1 << 5)) != 0;
    }
    return false;
#endif
}

/*
 * Find Pattern - Main Entry Point
 */
std::vector<qint64> PatternMatcher::findPattern(const QByteArray& data, const QByteArray& pattern,
                                               qint64 baseOffset)
{
    if (pattern.isEmpty() || data.isEmpty()) {
        return {};
    }
    
    // Choose optimal search method based on pattern size and CPU features
    if (pattern.size() == 1) {
        // Single byte search - use memchr for optimal performance
        std::vector<qint64> results;
        const char* dataPtr = data.constData();
        const char* found = dataPtr;
        const char target = pattern[0];
        
        while ((found = static_cast<const char*>(memchr(found, target, data.size() - (found - dataPtr)))) != nullptr) {
            results.push_back(baseOffset + (found - dataPtr));
            found++;
        }
        
        return results;
    }
    
    // Multi-byte pattern search
    if (m_useAVX2 && pattern.size() <= 32) {
        return findPatternAVX2(data, pattern, baseOffset);
    } else if (m_useSSE2 && pattern.size() <= 16) {
        return findPatternSSE2(data, pattern, baseOffset);
    } else {
        // Fallback to Boyer-Moore or KMP for longer patterns
        return findPatternBoyerMoore(data, pattern, baseOffset);
    }
}

/*
 * SSE2-Optimized Pattern Search
 */
std::vector<qint64> PatternMatcher::findPatternSSE2(const QByteArray& data, const QByteArray& pattern,
                                                   qint64 baseOffset)
{
    std::vector<qint64> results;
    
    if (pattern.size() > 16 || !m_useSSE2) {
        return findPatternBoyerMoore(data, pattern, baseOffset);
    }
    
    const char* dataPtr = data.constData();
    const char* patternPtr = pattern.constData();
    const qint64 dataSize = data.size();
    const qint64 patternSize = pattern.size();
    
    if (patternSize > dataSize) {
        return results;
    }
    
    // Create SSE2 pattern vector
    __m128i patternVec;
    if (patternSize >= 16) {
        patternVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(patternPtr));
    } else {
        // Pad smaller patterns
        char paddedPattern[16] = {0};
        memcpy(paddedPattern, patternPtr, patternSize);
        patternVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(paddedPattern));
    }
    
    // Search with SSE2 acceleration
    for (qint64 i = 0; i <= dataSize - patternSize; i += 16) {
        qint64 remaining = std::min(16LL, dataSize - i);
        
        if (remaining < patternSize) {
            // Handle remaining bytes with scalar comparison
            for (qint64 j = i; j <= dataSize - patternSize; ++j) {
                if (memcmp(dataPtr + j, patternPtr, patternSize) == 0) {
                    results.push_back(baseOffset + j);
                }
            }
            break;
        }
        
        __m128i dataVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dataPtr + i));
        __m128i cmpResult = _mm_cmpeq_epi8(dataVec, patternVec);
        
        int mask = _mm_movemask_epi8(cmpResult);
        
        // Check potential matches
        for (int bit = 0; bit < 16 && (i + bit) <= dataSize - patternSize; ++bit) {
            if (mask & (1 << bit)) {
                if (memcmp(dataPtr + i + bit, patternPtr, patternSize) == 0) {
                    results.push_back(baseOffset + i + bit);
                }
            }
        }
    }
    
    return results;
}

/*
 * AVX2-Optimized Pattern Search
 */
std::vector<qint64> PatternMatcher::findPatternAVX2(const QByteArray& data, const QByteArray& pattern,
                                                   qint64 baseOffset)
{
    std::vector<qint64> results;
    
#ifdef __AVX2__
    if (pattern.size() > 32 || !m_useAVX2) {
        return findPatternSSE2(data, pattern, baseOffset);
    }
    
    const char* dataPtr = data.constData();
    const char* patternPtr = pattern.constData();
    const qint64 dataSize = data.size();
    const qint64 patternSize = pattern.size();
    
    if (patternSize > dataSize) {
        return results;
    }
    
    // Create AVX2 pattern vector
    __m256i patternVec;
    if (patternSize >= 32) {
        patternVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(patternPtr));
    } else {
        // Pad smaller patterns
        char paddedPattern[32] = {0};
        memcpy(paddedPattern, patternPtr, patternSize);
        patternVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(paddedPattern));
    }
    
    // Search with AVX2 acceleration
    for (qint64 i = 0; i <= dataSize - patternSize; i += 32) {
        qint64 remaining = std::min(32LL, dataSize - i);
        
        if (remaining < patternSize) {
            // Handle remaining bytes with scalar comparison
            for (qint64 j = i; j <= dataSize - patternSize; ++j) {
                if (memcmp(dataPtr + j, patternPtr, patternSize) == 0) {
                    results.push_back(baseOffset + j);
                }
            }
            break;
        }
        
        __m256i dataVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dataPtr + i));
        __m256i cmpResult = _mm256_cmpeq_epi8(dataVec, patternVec);
        
        int mask = _mm256_movemask_epi8(cmpResult);
        
        // Check potential matches
        for (int bit = 0; bit < 32 && (i + bit) <= dataSize - patternSize; ++bit) {
            if (mask & (1 << bit)) {
                if (memcmp(dataPtr + i + bit, patternPtr, patternSize) == 0) {
                    results.push_back(baseOffset + i + bit);
                }
            }
        }
    }
#else
    // Fallback to SSE2 if AVX2 not available
    return findPatternSSE2(data, pattern, baseOffset);
#endif
    
    return results;
}

/*
 * Boyer-Moore Pattern Search (Fallback)
 */
std::vector<qint64> PatternMatcher::findPatternBoyerMoore(const QByteArray& data, const QByteArray& pattern,
                                                         qint64 baseOffset)
{
    std::vector<qint64> results;
    
    const char* dataPtr = data.constData();
    const char* patternPtr = pattern.constData();
    const qint64 dataSize = data.size();
    const qint64 patternSize = pattern.size();
    
    if (patternSize > dataSize || patternSize == 0) {
        return results;
    }
    
    // Build bad character table
    int badChar[256];
    for (int i = 0; i < 256; ++i) {
        badChar[i] = patternSize;
    }
    
    for (qint64 i = 0; i < patternSize - 1; ++i) {
        badChar[static_cast<unsigned char>(patternPtr[i])] = patternSize - 1 - i;
    }
    
    // Search using Boyer-Moore algorithm
    qint64 shift = 0;
    while (shift <= dataSize - patternSize) {
        qint64 j = patternSize - 1;
        
        while (j >= 0 && patternPtr[j] == dataPtr[shift + j]) {
            --j;
        }
        
        if (j < 0) {
            results.push_back(baseOffset + shift);
            shift += (shift + patternSize < dataSize) ? 
                    patternSize - badChar[static_cast<unsigned char>(dataPtr[shift + patternSize])] : 1;
        } else {
            shift += std::max(1LL, static_cast<qint64>(badChar[static_cast<unsigned char>(dataPtr[shift + j])] - patternSize + 1 + j));
        }
    }
    
    return results;
}

/*
 * Add Pattern to Aho-Corasick Automaton
 */
void PatternMatcher::addPattern(const QByteArray& pattern, int index)
{
    if (pattern.isEmpty()) {
        return;
    }
    
    if (index >= m_patterns.size()) {
        m_patterns.resize(index + 1);
    }
    m_patterns[index] = pattern;
    
    // Insert pattern into trie
    ACNode* current = m_root.get();
    for (char c : pattern) {
        if (current->children.find(c) == current->children.end()) {
            current->children[c] = std::make_unique<ACNode>();
            current->children[c]->depth = current->depth + 1;
        }
        current = current->children[c].get();
    }
    
    current->output.push_back(index);
    m_automatonBuilt = false;
}

/*
 * Build Aho-Corasick Automaton
 */
void PatternMatcher::buildAutomaton()
{
    if (m_automatonBuilt) {
        return;
    }
    
    buildFailureLinks();
    m_automatonBuilt = true;
    
    ForensicLogger::instance().info("automaton_built", "pattern_matcher",
                                   QStringLiteral("Built automaton for %1 patterns").arg(m_patterns.size()));
}

/*
 * Build Failure Links for Aho-Corasick
 */
void PatternMatcher::buildFailureLinks()
{
    std::queue<ACNode*> queue;
    
    // Initialize failure links for depth-1 nodes
    for (auto& [c, child] : m_root->children) {
        child->failure.reset(m_root.get());
        queue.push(child.get());
    }
    
    // Build failure links using BFS
    while (!queue.empty()) {
        ACNode* current = queue.front();
        queue.pop();
        
        for (auto& [c, child] : current->children) {
            queue.push(child.get());
            
            ACNode* temp = current->failure.get();
            while (temp != m_root.get() && temp->children.find(c) == temp->children.end()) {
                temp = temp->failure.get();
            }
            
            if (temp->children.find(c) != temp->children.end() && temp->children[c].get() != child.get()) {
                child->failure.reset(temp->children[c].get());
            } else {
                child->failure.reset(m_root.get());
            }
            
            // Merge output sets
            ACNode* failureNode = child->failure.get();
            child->output.insert(child->output.end(), failureNode->output.begin(), failureNode->output.end());
        }
    }
}

/*
 * Find All Patterns using Aho-Corasick
 */
std::vector<PatternMatcher::PatternMatch> PatternMatcher::findAllPatterns(const QByteArray& data, qint64 baseOffset)
{
    if (!m_automatonBuilt) {
        buildAutomaton();
    }
    
    return searchWithAutomaton(data, baseOffset);
}

/*
 * Search with Aho-Corasick Automaton
 */
std::vector<PatternMatcher::PatternMatch> PatternMatcher::searchWithAutomaton(const QByteArray& data, qint64 baseOffset)
{
    std::vector<PatternMatch> results;
    
    if (data.isEmpty() || !m_automatonBuilt) {
        return results;
    }
    
    ACNode* current = m_root.get();
    const char* dataPtr = data.constData();
    const qint64 dataSize = data.size();
    
    for (qint64 i = 0; i < dataSize; ++i) {
        char c = dataPtr[i];
        
        // Follow failure links until we find a match or reach root
        while (current != m_root.get() && current->children.find(c) == current->children.end()) {
            current = current->failure.get();
        }
        
        // Move to next state if possible
        if (current->children.find(c) != current->children.end()) {
            current = current->children[c].get();
        }
        
        // Check for pattern matches
        for (int patternIndex : current->output) {
            if (patternIndex < m_patterns.size()) {
                PatternMatch match;
                match.offset = baseOffset + i - m_patterns[patternIndex].size() + 1;
                match.patternIndex = patternIndex;
                match.pattern = m_patterns[patternIndex];
                results.push_back(match);
            }
        }
    }
    
    return results;
}

/*
 * Clear Patterns
 */
void PatternMatcher::clear()
{
    m_root = std::make_unique<ACNode>();
    m_patterns.clear();
    m_automatonBuilt = false;
}

} // namespace PhoenixDRS