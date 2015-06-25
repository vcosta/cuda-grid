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
#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <iostream>
#include <iomanip>

class Benchmark
{
public:
	static double get_time();

private:
	double timeCounter;

	void timer_start()
	{
		timeCounter = get_time();
	}
	double timer_stop()
	{
		timeCounter = get_time() - timeCounter;
		return timeCounter;
	}

public:
	Benchmark()
	{
	}

	void start(const std::string& name)
	{
		std::cout << "\n\n" << name << std::endl;
		timer_start();
	}
	void stop()
	{
		double benchmark = timer_stop();
		std::cout << " ok = " << benchmark << " seconds\n";
	}
	void stop_fps()
	{
		double benchmark = timer_stop();
		std::cout << " ok = " << benchmark << " seconds (i.e. " << 1.0/benchmark << " fps)\n";
	}
};


extern	Benchmark	bench;
#endif
