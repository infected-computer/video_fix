#include "include/PerformanceMonitor.h"

PerformanceMonitor::PerformanceMonitor(QObject *parent, int intervalMs)
    : QObject(parent)
{
    m_timer.setInterval(intervalMs);
    connect(&m_timer, &QTimer::timeout, this, &PerformanceMonitor::captureMetrics);
    init();
}

PerformanceMonitor::~PerformanceMonitor()
{
    stop();
}

void PerformanceMonitor::start()
{
    if (!m_timer.isActive()) {
        m_timer.start();
    }
}

void PerformanceMonitor::stop()
{
    if (m_timer.isActive()) {
        m_timer.stop();
    }
}

void PerformanceMonitor::captureMetrics()
{
    emit performanceUpdate(getCurrentCpuLoad(), getCurrentMemoryUsageMB());
}

#ifdef Q_OS_WIN
void PerformanceMonitor::init()
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    m_numProcessors = sysInfo.dwNumberOfProcessors;

    FILETIME ftime, fsys, fuser;
    GetSystemTimeAsFileTime(&ftime);
    memcpy(&m_lastCpu, &ftime, sizeof(FILETIME));

    HANDLE self = GetCurrentProcess();
    GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&m_lastSysCpu, &fsys, sizeof(FILETIME));
    memcpy(&m_lastUserCpu, &fuser, sizeof(FILETIME));
}

double PerformanceMonitor::getCurrentCpuLoad()
{
    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    double percent;

    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));

    HANDLE self = GetCurrentProcess();
    GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));

    percent = static_cast<double>((sys.QuadPart - m_lastSysCpu.QuadPart) + (user.QuadPart - m_lastUserCpu.QuadPart));
    percent /= static_cast<double>(now.QuadPart - m_lastCpu.QuadPart);
    percent /= m_numProcessors;

    m_lastCpu = now;
    m_lastUserCpu = user;
    m_lastSysCpu = sys;

    return percent * 100.0;
}

double PerformanceMonitor::getCurrentMemoryUsageMB()
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
}

#elif defined(Q_OS_LINUX)
void PerformanceMonitor::init()
{
    FILE* file = fopen("/proc/cpuinfo", "r");
    if (file) {
        char buffer[128];
        m_numProcessors = 0;
        while(fgets(buffer, sizeof(buffer), file)) {
            if (strncmp(buffer, "processor", 9) == 0) {
                m_numProcessors++;
            }
        }
        fclose(file);
    } else {
        m_numProcessors = 1;
    }

    struct tms timeSample;
    m_lastCpu = times(&timeSample);
    m_lastSysCpu = timeSample.tms_stime;
    m_lastUserCpu = timeSample.tms_utime;
}

double PerformanceMonitor::getCurrentCpuLoad()
{
    struct tms timeSample;
    clock_t now = times(&timeSample);
    if (now <= m_lastCpu || m_numProcessors == 0) {
        return 0.0;
    }

    double percent = static_cast<double>(timeSample.tms_stime - m_lastSysCpu + timeSample.tms_utime - m_lastUserCpu);
    percent /= static_cast<double>(now - m_lastCpu);
    percent /= m_numProcessors;

    m_lastCpu = now;
    m_lastSysCpu = timeSample.tms_stime;
    m_lastUserCpu = timeSample.tms_utime;

    return percent * 100.0;
}

double PerformanceMonitor::getCurrentMemoryUsageMB()
{
    FILE* file = fopen("/proc/self/status", "r");
    if (file) {
        long rss = 0;
        char line[128];
        while (fgets(line, sizeof(line), file) != nullptr) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line + 7, "%ld", &rss);
                break;
            }
        }
        fclose(file);
        return static_cast<double>(rss) / 1024.0;
    }
    return 0.0;
}

#elif defined(Q_OS_MACOS)
void PerformanceMonitor::init()
{
    m_totalTime = 0;
    m_prevTotalTime = 0;
    m_userTime = 0;
    m_prevUserTime = 0;
    m_systemTime = 0;
    m_prevSystemTime = 0;
}

double PerformanceMonitor::getCurrentCpuLoad()
{
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cpuinfo, &count) != KERN_SUCCESS) {
        return 0.0;
    }

    m_userTime = cpuinfo.cpu_ticks[CPU_STATE_USER];
    m_systemTime = cpuinfo.cpu_ticks[CPU_STATE_SYSTEM] + cpuinfo.cpu_ticks[CPU_STATE_NICE];
    m_totalTime = m_userTime + m_systemTime + cpuinfo.cpu_ticks[CPU_STATE_IDLE];

    double cpuLoad = 0.0;
    if (m_prevTotalTime > 0) {
        uint64_t total_delta = m_totalTime - m_prevTotalTime;
        uint64_t busy_delta = (m_userTime - m_prevUserTime) + (m_systemTime - m_prevSystemTime);
        if (total_delta > 0) {
            cpuLoad = static_cast<double>(busy_delta) / static_cast<double>(total_delta);
        }
    }

    m_prevTotalTime = m_totalTime;
    m_prevUserTime = m_userTime;
    m_prevSystemTime = m_systemTime;

    return cpuLoad * 100.0;
}

double PerformanceMonitor::getCurrentMemoryUsageMB()
{
    mach_task_basic_info_data_t taskInfo;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&taskInfo, &infoCount) == KERN_SUCCESS) {
        return static_cast<double>(taskInfo.resident_size) / (1024.0 * 1024.0);
    }
    return 0.0;
}
#else
// Fallback for unsupported platforms
void PerformanceMonitor::init() {}
double PerformanceMonitor::getCurrentCpuLoad() { return 0.0; }
double PerformanceMonitor::getCurrentMemoryUsageMB() { return 0.0; }
#endif
