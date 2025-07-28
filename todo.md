# מסמך דרישות למפתחים - PhoenixDRS Video Repair

## תקציר מנהלים
פרויקט PhoenixDRS דורש תיקונים קריטיים ומימוש של פונקציונליות בסיסית. מסמך זה מגדיר את המשימות הנדרשות בסדר עדיפות ברור.

---

## 🚨 Priority 1: תיקונים קריטיים (Sprint 1)

### 1.1 Memory Safety

#### דרישות:
- **החלף את כל ה-raw pointers ל-smart pointers**
- **הוסף RAII wrappers לכל ה-C APIs**
- **תקן buffer overflows בפרסינג**

#### משימות ספציפיות:

**Task 1.1.1: FFmpeg Memory Management**
```cpp
// קובץ: src/cpp/src/AdvancedVideoRepair/AdvancedVideoRepairEngine.cpp
// שורות: 145-289

// במקום:
AVFormatContext* ctx = nullptr;
// ... 
avformat_free_context(ctx);  // might not be reached

// לממש:
class AVFormatContextPtr {
    AVFormatContext* ctx = nullptr;
public:
    ~AVFormatContextPtr() { 
        if (ctx) avformat_close_input(&ctx); 
    }
    // Add move semantics, delete copy
};
```

**Acceptance Criteria:**
- [ ] אפס דליפות זיכרון ב-Valgrind
- [ ] כל ה-FFmpeg contexts עטופים ב-RAII
- [ ] AddressSanitizer עובר בלי שגיאות

**Task 1.1.2: CUDA Memory Management**
```cpp
// קובץ: src/cpp/src/AdvancedVideoRepair/cuda_kernels.cu
// יש ליצור: src/cpp/include/AdvancedVideoRepair/CudaUtils.h

// ליצור RAII wrapper:
template<typename T>
class CudaDeviceBuffer {
    T* d_ptr = nullptr;
    size_t count = 0;
public:
    explicit CudaDeviceBuffer(size_t n);
    ~CudaDeviceBuffer();
    // Implement rule of 5
};
```

**Acceptance Criteria:**
- [ ] כל ה-cudaMalloc עטופים ב-RAII
- [ ] cuda-memcheck לא מדווח על דליפות
- [ ] Unit tests לכל wrapper

---

### 1.2 Thread Safety

#### דרישות:
- **תקן את כל ה-data races**
- **הוסף proper synchronization**
- **השתמש ב-thread-safe containers**

#### משימות ספציפיות:

**Task 1.2.1: Frame Buffer Thread Safety**
```cpp
// קובץ: src/cpp/src/AdvancedVideoRepair/FrameReconstructor.cpp
// שורות: 78-134

// Problems to fix:
// 1. Shared frame_buffer without mutex
// 2. OpenMP loops modifying shared data
// 3. Non-atomic progress counters
```

**Implementation Requirements:**
```cpp
class ThreadSafeFrameBuffer {
    mutable std::shared_mutex mutex_;
    std::vector<cv::Mat> frames_;
    
public:
    void push_frame(cv::Mat frame) {
        std::unique_lock lock(mutex_);
        frames_.push_back(std::move(frame));
    }
    
    cv::Mat get_frame(size_t idx) const {
        std::shared_lock lock(mutex_);
        return frames_.at(idx).clone();
    }
};
```

**Acceptance Criteria:**
- [ ] ThreadSanitizer לא מוצא בעיות
- [ ] Stress test עם 16 threads עובר
- [ ] Performance degradation < 5%

---

### 1.3 מת ופישוט (2 ימים)

#### דרישות:
- **מחק את כל ה-over-engineered systems**
- **הסר קוד לא בשימוש**
- **פשט את הארכיטקטורה**

#### קבצים למחיקה מיידית:
```bash
# מחק את הקבצים הבאים:
src/cpp/src/AdvancedVideoRepair/AdvancedRobustnessSystem.cpp
src/cpp/src/AdvancedVideoRepair/AdvancedRobustnessSystem_Part2.cpp
src/cpp/include/AdvancedVideoRepair/CircuitBreaker.h
src/cpp/src/Core/LegacySupport/*
```

#### קוד לפישוט:
- הסר את כל ה-Circuit Breaker references
- הסר את ה-Resource Guard המיותר
- פשט את ה-error handling ל-try/catch רגיל

**Acceptance Criteria:**
- [ ] קוד קומפיילציה < 30 שניות
- [ ] LOC ירד ב-30% לפחות
- [ ] אין TODOs משנת 2023

---

## 📝 Priority 2: מימוש פונקציות Core (Sprint 2)

### 2.1 מימוש repair_video_file

#### דרישה:
מימוש מלא של הפונקציה הראשית לתיקון וידאו

**Task 2.1.1: Basic MP4 Repair**
```cpp
// קובץ: src/cpp/src/AdvancedVideoRepair/AdvancedVideoRepairEngine.cpp
// פונקציה: repair_video_file()

bool AdvancedVideoRepairEngine::repair_video_file(
    const std::string& input_file,
    const std::string& output_file,
    const RepairStrategy& strategy) {
    
    // Required implementation:
    // 1. Analyze file corruption
    auto analysis = analyze_corruption(input_file);
    
    // 2. Select repair method based on corruption type
    switch (analysis.primary_corruption) {
        case CorruptionType::HEADER_DAMAGE:
            return repair_header_corruption(input_file, output_file);
            
        case CorruptionType::MISSING_MOOV:
            return reconstruct_moov_atom(input_file, output_file);
            
        case CorruptionType::INDEX_CORRUPTION:
            return rebuild_index(input_file, output_file);
            
        default:
            return attempt_generic_repair(input_file, output_file);
    }
}
```

**Acceptance Criteria:**
- [ ] תיקון MP4 עם header פגום - עובד
- [ ] תיקון MP4 עם moov atom חסר - עובד
- [ ] תיקון MP4 עם index פגום - עובד
- [ ] 10 test videos עוברים בהצלחה

---

### 2.2 מימוש analyze_corruption

**Task 2.2.1: Corruption Detection**
```cpp
// קובץ: src/cpp/src/AdvancedVideoRepair/ContainerAnalyzer.cpp

CorruptionAnalysis analyze_corruption(const std::string& file_path) {
    CorruptionAnalysis result;
    
    // 1. Check file header
    if (!validate_file_header(file_path)) {
        result.add_issue(CorruptionType::HEADER_DAMAGE);
    }
    
    // 2. Parse container structure
    auto structure = parse_container_structure(file_path);
    
    // 3. Validate essential atoms/boxes
    if (!structure.has_moov()) {
        result.add_issue(CorruptionType::MISSING_MOOV);
    }
    
    // 4. Check stream integrity
    validate_stream_integrity(structure, result);
    
    return result;
}
```

**Test Cases Required:**
```cpp
TEST(CorruptionAnalysis, DetectsHeaderCorruption) {
    auto result = analyze_corruption("test_data/corrupted_header.mp4");
    EXPECT_TRUE(result.has_issue(CorruptionType::HEADER_DAMAGE));
}

TEST(CorruptionAnalysis, DetectsMissingMoov) {
    auto result = analyze_corruption("test_data/no_moov.mp4");
    EXPECT_TRUE(result.has_issue(CorruptionType::MISSING_MOOV));
}
// At least 8 more test cases
```

---

### 2.3 מימוש Frame Reconstruction

**Task 2.3.1: Basic Frame Interpolation**
```cpp
// קובץ: src/cpp/src/AdvancedVideoRepair/FrameReconstructor.cpp

bool reconstruct_missing_frame(
    const cv::Mat& prev_frame,
    const cv::Mat& next_frame,
    cv::Mat& output_frame) {
    
    // Basic implementation first:
    // 1. Simple linear interpolation
    cv::addWeighted(prev_frame, 0.5, next_frame, 0.5, 0, output_frame);
    
    // 2. TODO in next sprint: motion compensation
    
    return true;
}
```

**Acceptance Criteria:**
- [ ] יכולת לשחזר פריים בודד חסר
- [ ] יכולת לשחזר עד 5 פריימים רצופים
- [ ] PSNR > 30dB לפריימים משוחזרים

---

## 🧪 Priority 3: Testing & Documentation (Sprint 3)

### 3.1 Unit Tests

#### דרישה מינימלית: 60% code coverage

**Required Test Files:**
```
tests/
├── test_video_repair_engine.cpp    (20+ tests)
├── test_container_analyzer.cpp      (15+ tests)
├── test_frame_reconstructor.cpp     (15+ tests)
├── test_ffmpeg_integration.cpp      (10+ tests)
└── test_cuda_kernels.cu            (10+ tests)
```

**Test Template:**
```cpp
class VideoRepairEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<VideoRepairEngine>();
        engine->initialize();
    }
    
    void TearDown() override {
        engine->shutdown();
    }
    
    std::unique_ptr<VideoRepairEngine> engine;
};

TEST_F(VideoRepairEngineTest, RepairsSimpleHeaderCorruption) {
    // Arrange
    const std::string corrupted = "test_data/header_corrupted.mp4";
    const std::string output = "test_output/fixed.mp4";
    
    // Act
    auto result = engine->repair_video_file(corrupted, output);
    
    // Assert
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(std::filesystem::exists(output));
    EXPECT_TRUE(can_play_video(output));
}
```

---

### 3.2 Integration Tests

**Required Test Scenarios:**
1. תיקון קובץ 1GB
2. תיקון מקביל של 10 קבצים
3. תיקון עם GPU enabled/disabled
4. Python binding tests

---

### 3.3 Documentation

**Required Documentation:**

**1. API Documentation**
```cpp
/**
 * @brief Repairs a corrupted video file
 * 
 * @param input_file Path to corrupted video file
 * @param output_file Path where repaired video will be saved
 * @param strategy Repair strategy configuration
 * 
 * @return RepairResult containing success status and metrics
 * 
 * @throws std::runtime_error if input file cannot be opened
 * @throws std::bad_alloc if insufficient memory
 * 
 * @example
 * VideoRepairEngine engine;
 * auto result = engine.repair_video_file("corrupted.mp4", "fixed.mp4");
 * if (result.success) {
 *     std::cout << "Repair successful!" << std::endl;
 * }
 */
```

**2. README.md Update**
- מחק claims על features שלא קיימים
- הוסף דוגמאות עבודה אמיתיות
- הוסף troubleshooting section

---

## 📊 Metrics & Success Criteria

### Sprint 1 Success Metrics:
- [ ] **Zero crashes** בכל הבדיקות
- [ ] **Zero memory leaks** (Valgrind clean)
- [ ] **Compilation time < 1 minute**
- [ ] **All compiler warnings fixed**

### Sprint 2 Success Metrics:
- [ ] **Basic MP4 repair working** (5 test cases)
- [ ] **Performance**: יכול לתקן 100MB תוך פחות מ-10 שניות
- [ ] **Python binding functional**
- [ ] **CLI tool working**

### Sprint 3 Success Metrics:
- [ ] **Code coverage > 60%**
- [ ] **All tests passing**
- [ ] **Documentation complete**
- [ ] **Demo video created**

---

## 🛠️ Development Environment Setup

### Required Tools:
```bash
# C++ Development
- CMake 3.20+
- C++17 compliant compiler
- FFmpeg 4.4+ development libraries
- OpenCV 4.5+
- CUDA Toolkit 11.0+ (optional)

# Testing
- Google Test
- Valgrind
- AddressSanitizer
- ThreadSanitizer

# Code Quality
- clang-format
- clang-tidy
- cppcheck
```

### Build Instructions:
```bash
# Debug build with sanitizers
mkdir build_debug && cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DENABLE_ASAN=ON \
         -DENABLE_TSAN=ON
make -j8

# Run tests
ctest --output-on-failure

# Check for memory leaks
valgrind --leak-check=full ./tests/all_tests
```

---

## TODOs

Memory safety
Thread safety  
Code cleanup

repair_video_file
analyze_corruption
frame_reconstruction

Unit tests
Integration tests
Documentation
Final review

---

## ⚠️ Important Notes

1. **אל תוסיפו features חדשים** עד שהבסיס עובד
2. **כל קוד חדש חייב unit test**
3. **Code review חובה** לכל PR
4. **Daily standup** לעדכון סטטוס
5. **אפס tolerance** ל-compiler warnings

---

## 📞 Contact & Support

**Technical Lead:** [Your Name]
**Slack Channel:** #phoenixdrs-dev
**Daily Standup:** 09:00 IDT
**Code Review SLA:** 4 hours

**בהצלחה! 🚀**