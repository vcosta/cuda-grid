/*****************************************************************************
Copyright (c) 2015 Vasco Alexandre da Silva Costa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. See the MIT License for more details.
*****************************************************************************/
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

