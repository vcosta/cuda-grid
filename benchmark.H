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
