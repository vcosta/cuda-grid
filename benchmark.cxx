#include "benchmark.H"

#ifdef  __unix__
#define HAVE_GETTIMEOFDAY
#include <sys/time.h>
#include <unistd.h>
#endif
/*
#ifdef _WIN32
#define HAVE_TIMEGETTIME
#include <windows.h>
#endif
*/
#include <ctime>

Benchmark bench;


double Benchmark::get_time() 
{
#if defined(_POSIX_TIMERS)
	struct timespec ts;

	clock_gettime(CLOCK_REALTIME, &ts);
	return (double(ts.tv_sec) + double(ts.tv_nsec)*1e-9);
#elif defined(HAVE_GETTIMEOFDAY)
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (double(tv.tv_sec) + double(tv.tv_usec)*1e-6);
#elif defined(HAVE_TIMEGETTIME)
	return (double(timeGetTime()*1e-3));
#else
	return (double(clock())/CLOCKS_PER_SEC);
#endif
}

