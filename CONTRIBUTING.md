# Contributing to PhoenixDRS Professional

First off, thank you for considering contributing to PhoenixDRS Professional! It's people like you that make this project a great tool for video recovery and data forensics.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Pledge

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Inclusive**: Welcome developers of all backgrounds and experience levels
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Professional**: Maintain a professional demeanor in all interactions

## Getting Started

### What Can You Contribute?

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new capabilities
- **📝 Documentation**: Improve guides, tutorials, and API docs
- **🧪 Testing**: Write tests or test new features
- **🔧 Code**: Fix bugs or implement new features
- **🎨 UI/UX**: Improve the user interface and experience
- **🌐 Translations**: Help localize the application

### Areas Needing Help

- AI model optimization and new model integration
- Professional video format support (MXF, IMF, etc.)
- Cross-platform compatibility improvements
- Performance optimization for large files
- Accessibility improvements
- Documentation and tutorials

## Development Setup

### Prerequisites

- Python 3.9+ with pip
- Node.js 16+ with npm
- CMake 3.20+ (for C++ components)
- Git
- A C++ compiler (Visual Studio 2022 on Windows, GCC/Clang on Linux/macOS)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/phoenixdrs.git
   cd phoenixdrs
   ```

3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/phoenixdrs.git
   ```

### Environment Setup

#### Automated Setup
```bash
# Windows
.\setup.bat

# Linux/macOS
chmod +x setup.sh && ./setup.sh
```

#### Manual Setup
```bash
# Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Node.js dependencies
npm install

# Build C++ components
cd src/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON
make -j$(nproc)  # On Windows: cmake --build . --parallel
cd ../..

# Build desktop app
npm run build
```

### Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit regularly with clear messages

3. Keep your branch updated with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. Push your changes and create a pull request

## How to Contribute

### 🐛 Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**Use this template for bug reports:**

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g., Windows 11, Ubuntu 22.04]
 - PhoenixDRS Version: [e.g., 2.0.0]
 - Python Version: [e.g., 3.11.0]
 - Node.js Version: [e.g., 18.17.0]

**Additional Context**
Add any other context about the problem here.
```

### ✨ Suggesting Features

Feature requests are welcome! Please provide:

1. **Clear description** of the feature
2. **Use case** - why would this be useful?
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**

### 📝 Improving Documentation

Documentation improvements are always appreciated:

- Fix typos or clarify existing content
- Add missing documentation for features
- Create tutorials or examples
- Improve API documentation
- Add translations

## Coding Standards

### Python Code Style

Follow PEP 8 with these specific guidelines:

```python
# Use type hints
def process_video(file_path: str, options: Dict[str, Any]) -> VideoResult:
    """
    Process a video file with specified options.
    
    Args:
        file_path: Path to the input video file
        options: Processing options dictionary
        
    Returns:
        VideoResult object containing processing results
        
    Raises:
        VideoProcessingError: If processing fails
    """
    pass

# Use dataclasses for structured data
@dataclass
class VideoMetadata:
    duration: float
    width: int
    height: int
    codec: str
    
# Use async/await for I/O operations
async def analyze_video_async(file_path: str) -> VideoAnalysis:
    async with aiofiles.open(file_path, 'rb') as file:
        data = await file.read(1024)
        return analyze_header(data)
```

### TypeScript/JavaScript Code Style

Follow Airbnb style guide with these additions:

```typescript
// Use strict typing
interface VideoConfig {
  inputPath: string;
  outputPath: string;
  quality: 'low' | 'medium' | 'high';
  enableAI: boolean;
}

// Use async/await over promises
const processVideo = async (config: VideoConfig): Promise<ProcessResult> => {
  try {
    const result = await videoEngine.process(config);
    return result;
  } catch (error) {
    logger.error('Video processing failed:', error);
    throw error;
  }
};

// Use proper error handling
class VideoProcessingError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'VideoProcessingError';
  }
}
```

### C++ Code Style

Follow Google C++ Style Guide:

```cpp
// Use modern C++17 features
class VideoProcessor {
public:
  explicit VideoProcessor(const VideoConfig& config) : config_(config) {}
  
  // Use smart pointers
  std::unique_ptr<ProcessResult> ProcessVideo(const std::string& input_path);
  
  // Use const correctness
  [[nodiscard]] bool IsInitialized() const noexcept { return initialized_; }
  
private:
  VideoConfig config_;
  bool initialized_ = false;
  
  // Use RAII for resource management
  std::unique_ptr<VideoDecoder> decoder_;
};

// Use structured bindings
auto [success, result] = processor.ProcessVideo("input.mp4");
if (success) {
  // Handle success
}

// Use std::optional for optional values
std::optional<VideoMetadata> ExtractMetadata(const std::string& file_path);
```

### Commit Message Format

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(ai): add RIFE frame interpolation model

Implements RIFE (Real-Time Intermediate Flow Estimation) for frame
interpolation in video repair workflows. Includes GPU acceleration
and configurable quality settings.

Closes #123
```

```
fix(cpp): resolve memory leak in video decoder

- Fix improper cleanup in VideoDecoder destructor
- Add RAII wrapper for FFmpeg contexts
- Update unit tests to verify resource cleanup

Fixes #456
```

## Testing Guidelines

### Python Tests

```python
import pytest
from unittest.mock import Mock, patch
from phoenixdrs import VideoRepairEngine

class TestVideoRepairEngine:
    @pytest.fixture
    def engine(self):
        return VideoRepairEngine()
    
    @pytest.mark.asyncio
    async def test_video_analysis(self, engine):
        """Test video file analysis functionality."""
        # Arrange
        test_file = "tests/data/test_video.mp4"
        
        # Act
        result = await engine.analyze_video(test_file)
        
        # Assert
        assert result.success
        assert result.duration > 0
        assert result.width > 0
        assert result.height > 0
    
    def test_configuration_validation(self, engine):
        """Test configuration parameter validation."""
        with pytest.raises(ValueError, match="Invalid quality factor"):
            engine.configure(quality_factor=1.5)  # Should be 0.0-1.0
```

### C++ Tests

```cpp
#include <gtest/gtest.h>
#include "video_repair_engine.h"

class VideoRepairEngineTest : public ::testing::Test {
protected:
  void SetUp() override {
    engine_ = std::make_unique<VideoRepairEngine>();
  }
  
  std::unique_ptr<VideoRepairEngine> engine_;
};

TEST_F(VideoRepairEngineTest, InitializationSuccess) {
  EXPECT_TRUE(engine_->Initialize());
  EXPECT_TRUE(engine_->IsInitialized());
}

TEST_F(VideoRepairEngineTest, ProcessValidVideo) {
  ASSERT_TRUE(engine_->Initialize());
  
  const std::string input_path = "tests/data/test_video.mp4";
  const std::string output_path = "tests/output/repaired_video.mp4";
  
  auto result = engine_->ProcessVideo(input_path, output_path);
  
  EXPECT_TRUE(result.success);
  EXPECT_GT(result.frames_processed, 0);
}
```

### Integration Tests

```typescript
import { test, expect } from '@playwright/test';

test.describe('Video Repair Workflow', () => {
  test('complete repair workflow', async ({ page }) => {
    // Navigate to application
    await page.goto('/');
    
    // Upload test video
    await page.setInputFiles('input[type="file"]', 'tests/data/corrupted_video.mp4');
    
    // Configure repair settings
    await page.selectOption('[data-testid="quality-select"]', 'high');
    await page.check('[data-testid="enable-ai-checkbox"]');
    
    // Start repair process
    await page.click('[data-testid="start-repair-button"]');
    
    // Wait for completion
    await expect(page.locator('[data-testid="progress-bar"]')).toHaveAttribute('value', '100');
    
    // Verify results
    await expect(page.locator('[data-testid="repair-status"]')).toHaveText('Completed Successfully');
  });
});
```

### Running Tests

```bash
# Python tests
pytest tests/ --cov=src --cov-report=html

# C++ tests  
cd src/cpp/build
ctest --verbose

# JavaScript/TypeScript tests
npm test

# Integration tests
npm run test:e2e

# All tests
npm run test:all
```

## Submitting Changes

### Pull Request Process

1. **Create a descriptive title** that summarizes the change
2. **Fill out the PR template** completely
3. **Link related issues** using "Closes #123" or "Fixes #456"
4. **Ensure all tests pass** and add new tests for new functionality
5. **Update documentation** if needed
6. **Request review** from appropriate maintainers

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented appropriately
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated checks** must pass (CI/CD, linting, tests)
2. **Code review** by at least one maintainer
3. **Testing** on different platforms if applicable
4. **Documentation review** if docs were changed
5. **Final approval** and merge

## Performance Considerations

### Code Performance

- **Profile before optimizing**: Use profiling tools to identify bottlenecks
- **Memory efficiency**: Minimize memory allocations in hot paths
- **Algorithmic complexity**: Consider Big O notation for algorithms
- **GPU utilization**: Leverage GPU acceleration where appropriate

### Testing Performance

```python
# Benchmark critical functions
@pytest.mark.benchmark(group="video_processing")
def test_video_processing_performance(benchmark):
    result = benchmark(process_video, "test_input.mp4")
    assert result.success
```

## Security Guidelines

### Security Best Practices

- **Input validation**: Always validate user inputs
- **Path traversal**: Prevent directory traversal attacks
- **Memory safety**: Use safe string handling in C++
- **Dependency scanning**: Keep dependencies updated
- **Secrets management**: Never commit secrets or API keys

### Reporting Security Issues

**DO NOT** create public issues for security vulnerabilities. Instead:

1. Email security@phoenixdrs.com with details
2. Include proof of concept if applicable
3. Allow time for patching before public disclosure

## Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Discord Server**: Real-time chat with other contributors
- **Stack Overflow**: Tag questions with `phoenixdrs`

### Maintainer Responsibilities

Maintainers are responsible for:

- Reviewing and merging pull requests
- Triaging and labeling issues
- Maintaining code quality standards
- Coordinating releases
- Community management

Thank you for contributing to PhoenixDRS Professional! Your efforts help make digital forensics and video recovery more accessible to everyone. 🚀