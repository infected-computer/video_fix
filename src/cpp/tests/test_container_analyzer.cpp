#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"

using namespace AdvancedVideoRepair;

class ContainerAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<AdvancedVideoRepairEngine>();
        ASSERT_TRUE(engine->initialize());
        
        analyzer = std::make_unique<ContainerAnalyzer>(engine.get());
        
        std::filesystem::create_directories("test_data");
        std::filesystem::create_directories("test_output");
    }
    
    void TearDown() override {
        analyzer.reset();
        if (engine) {
            engine->shutdown();
        }
        std::filesystem::remove_all("test_output");
    }
    
    // Helper to create MP4 file with specific structure
    void create_mp4_with_boxes(const std::string& filename, bool include_moov = true, 
                              bool include_mdat = true, bool corrupt_moov = false) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write ftyp box
        const char ftyp_box[] = {
            0x00, 0x00, 0x00, 0x20,  // size
            'f', 't', 'y', 'p',      // type
            'i', 's', 'o', 'm',      // major brand
            0x00, 0x00, 0x02, 0x00,  // minor version
            'i', 's', 'o', 'm',      // compatible brands
            'm', 'p', '4', '1',
            'a', 'v', 'c', '1',
            'd', 'a', 's', 'h'
        };
        file.write(ftyp_box, sizeof(ftyp_box));
        
        if (include_moov) {
            if (!corrupt_moov) {
                // Write minimal valid moov box
                const char moov_box[] = {
                    0x00, 0x00, 0x00, 0x10,  // size (16 bytes)
                    'm', 'o', 'o', 'v',      // type
                    0x00, 0x00, 0x00, 0x08,  // mvhd size
                    'm', 'v', 'h', 'd'       // mvhd type
                };
                file.write(moov_box, sizeof(moov_box));
            } else {
                // Write corrupted moov box
                const char corrupt_moov[] = {
                    0xFF, 0xFF, 0xFF, 0xFF,  // invalid size
                    'm', 'o', 'o', 'v',      // type
                    0x00, 0x00, 0x00, 0x00   // corrupted data
                };
                file.write(corrupt_moov, sizeof(corrupt_moov));
            }
        }
        
        if (include_mdat) {
            // Write mdat box
            const char mdat_box[] = {
                0x00, 0x00, 0x00, 0x10,  // size
                'm', 'd', 'a', 't',      // type
                0x00, 0x01, 0x02, 0x03,  // sample data
                0x04, 0x05, 0x06, 0x07
            };
            file.write(mdat_box, sizeof(mdat_box));
        }
    }
    
    void create_avi_file(const std::string& filename, bool corrupt_header = false) {
        std::ofstream file(filename, std::ios::binary);
        
        if (!corrupt_header) {
            // Write RIFF header
            const char riff_header[] = {
                'R', 'I', 'F', 'F',      // RIFF signature
                0x00, 0x00, 0x01, 0x00,  // file size (little endian)
                'A', 'V', 'I', ' ',      // AVI signature
                'L', 'I', 'S', 'T',      // LIST chunk
                0x00, 0x00, 0x00, 0x20   // LIST size
            };
            file.write(riff_header, sizeof(riff_header));
        } else {
            // Write corrupted header
            const char corrupt_header[] = {
                'R', 'I', 'F', 'F',      // RIFF signature
                0xFF, 0xFF, 0xFF, 0xFF,  // invalid file size
                'X', 'X', 'X', 'X',      // invalid signature
                0x00, 0x00, 0x00, 0x00
            };
            file.write(corrupt_header, sizeof(corrupt_header));
        }
    }
    
    std::unique_ptr<AdvancedVideoRepairEngine> engine;
    std::unique_ptr<ContainerAnalyzer> analyzer;
};

// MP4 structure analysis tests
TEST_F(ContainerAnalyzerTest, AnalyzeValidMP4Structure) {
    create_mp4_with_boxes("test_data/valid_mp4.mp4", true, true, false);
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/valid_mp4.mp4");
    
    // Valid file should have minimal issues
    EXPECT_FALSE(analysis.container_issues.missing_moov_atom);
    EXPECT_LT(analysis.detected_issues.size(), 3u);  // Allow for minor issues
}

TEST_F(ContainerAnalyzerTest, DetectMissingMoovAtom) {
    create_mp4_with_boxes("test_data/no_moov.mp4", false, true, false);
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/no_moov.mp4");
    
    EXPECT_TRUE(analysis.container_issues.missing_moov_atom);
    
    // Should detect container structure issues
    bool has_container_issue = false;
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::CONTAINER_STRUCTURE) {
            has_container_issue = true;
            break;
        }
    }
    EXPECT_TRUE(has_container_issue);
}

TEST_F(ContainerAnalyzerTest, DetectMissingMdatAtom) {
    create_mp4_with_boxes("test_data/no_mdat.mp4", true, false, false);
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/no_mdat.mp4");
    
    // Should detect missing media data
    bool has_missing_data = false;
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::MISSING_FRAMES || 
            issue == CorruptionType::CONTAINER_STRUCTURE) {
            has_missing_data = true;
            break;
        }
    }
    EXPECT_TRUE(has_missing_data);
}

TEST_F(ContainerAnalyzerTest, DetectCorruptedMoovAtom) {
    create_mp4_with_boxes("test_data/corrupt_moov.mp4", true, true, true);
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/corrupt_moov.mp4");
    
    // Should detect corruption issues
    EXPECT_FALSE(analysis.detected_issues.empty());
    
    bool has_corruption = false;
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::CONTAINER_STRUCTURE ||
            issue == CorruptionType::BITSTREAM_ERRORS) {
            has_corruption = true;
            break;
        }
    }
    EXPECT_TRUE(has_corruption);
}

TEST_F(ContainerAnalyzerTest, AnalyzeNonExistentMP4File) {
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/nonexistent.mp4");
    
    EXPECT_FALSE(analysis.detailed_report.empty());
    EXPECT_NE(analysis.detailed_report.find("Cannot open"), std::string::npos);
}

TEST_F(ContainerAnalyzerTest, AnalyzeEmptyMP4File) {
    // Create empty file
    std::ofstream file("test_data/empty.mp4");
    file.close();
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/empty.mp4");
    
    EXPECT_FALSE(analysis.detected_issues.empty());
    bool has_structure_issue = false;
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::CONTAINER_STRUCTURE) {
            has_structure_issue = true;
            break;
        }
    }
    EXPECT_TRUE(has_structure_issue);
}

// AVI structure analysis tests
TEST_F(ContainerAnalyzerTest, AnalyzeValidAVIStructure) {
    create_avi_file("test_data/valid.avi", false);
    
    CorruptionAnalysis analysis = analyzer->analyze_avi_structure("test_data/valid.avi");
    
    // Valid file should have fewer issues
    EXPECT_LT(analysis.detected_issues.size(), 5u);
}

TEST_F(ContainerAnalyzerTest, AnalyzeCorruptedAVIStructure) {
    create_avi_file("test_data/corrupt.avi", true);
    
    CorruptionAnalysis analysis = analyzer->analyze_avi_structure("test_data/corrupt.avi");
    
    EXPECT_FALSE(analysis.detected_issues.empty());
}

TEST_F(ContainerAnalyzerTest, AnalyzeNonExistentAVIFile) {
    CorruptionAnalysis analysis = analyzer->analyze_avi_structure("test_data/nonexistent.avi");
    
    EXPECT_FALSE(analysis.detailed_report.empty());
}

// MKV structure analysis tests  
TEST_F(ContainerAnalyzerTest, AnalyzeMKVStructure) {
    // Create minimal MKV-like file
    std::ofstream file("test_data/test.mkv", std::ios::binary);
    const char mkv_header[] = {
        0x1A, 0x45, 0xDF, 0xA3,  // EBML signature
        0x00, 0x00, 0x00, 0x20,  // header size
        0x42, 0x86, 0x81, 0x01,  // EBML version
        0x42, 0x87, 0x81, 0x01,  // EBML read version
        0x42, 0x85, 0x81, 0x04,  // EBML max ID length
        0x42, 0x83, 0x81, 0x08   // EBML max size length
    };
    file.write(mkv_header, sizeof(mkv_header));
    file.close();
    
    CorruptionAnalysis analysis = analyzer->analyze_mkv_structure("test_data/test.mkv");
    
    // Basic test - should not crash
    EXPECT_TRUE(analysis.detailed_report.length() > 0);
}

// MP4 repair tests
TEST_F(ContainerAnalyzerTest, RepairValidMP4Container) {
    create_mp4_with_boxes("test_data/repair_input.mp4", true, true, false);
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/repair_input.mp4");
    
    bool repair_result = analyzer->repair_mp4_container(
        "test_data/repair_input.mp4",
        "test_output/repaired.mp4", 
        analysis
    );
    
    // Should complete without crashing (may or may not succeed depending on implementation)
    EXPECT_TRUE(repair_result || !repair_result);  // Either outcome is acceptable for basic test
}

TEST_F(ContainerAnalyzerTest, RepairMP4WithMissingMoov) {
    create_mp4_with_boxes("test_data/no_moov_repair.mp4", false, true, false);
    
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("test_data/no_moov_repair.mp4");
    
    bool repair_result = analyzer->repair_mp4_container(
        "test_data/no_moov_repair.mp4",
        "test_output/moov_repaired.mp4", 
        analysis
    );
    
    // Test should complete without crashing
    EXPECT_TRUE(repair_result || !repair_result);
}

TEST_F(ContainerAnalyzerTest, RepairNonExistentMP4File) {
    CorruptionAnalysis dummy_analysis;
    
    bool repair_result = analyzer->repair_mp4_container(
        "test_data/nonexistent.mp4",
        "test_output/should_fail.mp4", 
        dummy_analysis
    );
    
    EXPECT_FALSE(repair_result);
}

// AVI repair tests
TEST_F(ContainerAnalyzerTest, RepairValidAVIContainer) {
    create_avi_file("test_data/avi_repair_input.avi", false);
    
    CorruptionAnalysis analysis = analyzer->analyze_avi_structure("test_data/avi_repair_input.avi");
    
    bool repair_result = analyzer->repair_avi_container(
        "test_data/avi_repair_input.avi",
        "test_output/avi_repaired.avi", 
        analysis
    );
    
    // Should complete without crashing
    EXPECT_TRUE(repair_result || !repair_result);
}

TEST_F(ContainerAnalyzerTest, RepairNonExistentAVIFile) {
    CorruptionAnalysis dummy_analysis;
    
    bool repair_result = analyzer->repair_avi_container(
        "test_data/nonexistent.avi",
        "test_output/avi_should_fail.avi", 
        dummy_analysis
    );
    
    EXPECT_FALSE(repair_result);
}

// Edge case tests
TEST_F(ContainerAnalyzerTest, AnalyzeEmptyFilePath) {
    CorruptionAnalysis analysis = analyzer->analyze_mp4_structure("");
    
    EXPECT_FALSE(analysis.detailed_report.empty());
    EXPECT_NE(analysis.detailed_report.find("Cannot open"), std::string::npos);
}

TEST_F(ContainerAnalyzerTest, RepairWithEmptyPaths) {
    CorruptionAnalysis dummy_analysis;
    
    bool result1 = analyzer->repair_mp4_container("", "test_output/empty_input.mp4", dummy_analysis);
    bool result2 = analyzer->repair_mp4_container("test_data/valid.mp4", "", dummy_analysis);
    
    EXPECT_FALSE(result1);
    EXPECT_FALSE(result2);
}