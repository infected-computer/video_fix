#include <gtest/gtest.h>
#include "include/ForensicLogger.h"

// Example test case
TEST(ExampleTest, AlwaysPasses) {
    EXPECT_EQ(1, 1);
}

// Example test for the ForensicLogger
TEST(ForensicLoggerTest, SingletonInstance) {
    ForensicLogger* instance1 = ForensicLogger::instance();
    ForensicLogger* instance2 = ForensicLogger::instance();
    ASSERT_NE(instance1, nullptr);
    ASSERT_EQ(instance1, instance2);
}
