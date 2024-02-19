#pragma once
#include "lib.h"

#if defined(__GNUC__)
#include <unistd.h>
#if defined(__linux__)
#include <linux/limits.h>
#include <unistd.h>
#endif
#endif

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/time.h>
#endif

namespace rlop {
   inline std::string GetDatetime(const std::string& format = "%Y%m%d_%H%M%S") {
      auto time_t = std::time(nullptr);
      struct std::tm* timeinfo = std::localtime(&time_t);
      char time_buf[80];
      std::strftime(time_buf, sizeof(time_buf), format.c_str(), timeinfo);
      return time_buf;
   }

#if defined(__GNUC__) && defined(__linux__)
   inline uint32_t GetPID() {
      return getpid();
   }

   inline uint32_t GetTickCount() {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return static_cast<uint32_t>((tv.tv_sec + tv.tv_usec) & 0xffffffff);
   }

   inline int64_t GetProcessMemoryUsage() {
      std::string path = "/proc/" + std::to_string(getpid()) + "/statm";
      std::ifstream file_stream(path);
      if (!file_stream)
         return 0;
      unsigned size = 0;
      file_stream >> size;
      if (!file_stream) 
         return 0;
      return static_cast<int64_t>(size) * 1024;
   }

   inline int64_t GetAvailableMemorySize() {
      std::ifstream file("/proc/meminfo");
      std::string line;
      int64_t available_bytes = 0;
      if (file.is_open()) {
         while (std::getline(file, line)) {
               if (line.substr(0, 14) == "MemAvailable: ") {
                  std::istringstream iss(line.substr(14));
                  iss >> available_bytes;
                  available_bytes *= 1024;
                  break;
               }
         }
         file.close();
      }
      return available_bytes;
   }

#elif defined(_MSC_VER)
   inline uint32_t GetPID() {
      return GetCurrentProcessId();
   }

   inline uint32_t GetSysTickCount() {
      return GetTickCount();
   }

   inline int64_t GetProcessMemoryUsage() {
      HANDLE hProcess;
      PROCESS_MEMORY_COUNTERS pmc;
      hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, GetCurrentProcessId());
      int64_t memory = 0;
      if (hProcess) {
         if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
            memory = pmc.WorkingSetSize;
         }
         CloseHandle(hProcess);
      }
      return memory;
   }
   
   inline int64_t GetAvailableMemoryBytes() {
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      GlobalMemoryStatusEx(&memInfo);
      DWORDLONG available_bytes = memInfo.ullAvailPhys;
      return available_bytes;
   }

#endif
}