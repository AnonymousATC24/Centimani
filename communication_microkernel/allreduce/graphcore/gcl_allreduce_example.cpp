// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include <bits/stdc++.h>
#include <sys/time.h>
using namespace std;
using namespace poplar;

poplar::Device attachToIPUs(int numDevices) {
  const auto dm = DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(TargetType::IPU, numDevices);
  if (hwDevices.empty()) {
    throw std::runtime_error("No such device exist");
  }
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    for (auto &d : hwDevices) {
      if (d.attach()) {
        return std::move(d);
      }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (std::chrono::steady_clock::now() - start > std::chrono::minutes(5)) {
      std::cerr << "Timeout\n";
      break;
    }
  }
  throw std::runtime_error("No available devices");
}

int main(int argc, char  **argv) {

int64_t all_reduce_kernels_size[] = {100000, 3097600, 4194304, 6553600, 16777217, 38360000, 64500000};
int64_t all_reduce_kernels_repeat[] = {10000, 10000, 10000, 10000, 1000, 100, 100};

  int numDevices = atoi(argv[1]);
  int replicationFactor = numDevices;
  const auto type = FLOAT;

  //std::cout << sizeof(type) << endl;

  struct timeval start, end;

  for(int loop=0; loop<=6; loop++){

  long unsigned int numElems = all_reduce_kernels_size[loop];

  // Acquire IPU device
  const auto device = attachToIPUs(numDevices);

  // Create graph and data variable
  Graph graph(device.getTarget(), replication_factor(replicationFactor));
  popops::addCodelets(graph);
  Tensor data = graph.addVariable(type, {numElems}, VariableMappingMethod::LINEAR);

  // Set up data streams:
  // DataStream inStream = graph.addHostToDeviceFIFO("in", type, numElems);
  // DataStream outStream = graph.addDeviceToHostFIFO("out", type, numElems);

  // Main program
  program::Sequence prog;
  gcl::allReduceInPlaceCrossReplica(graph, data, gcl::CollectiveOperator::ADD, prog);

  // On device repeat the program before returning control to CPU
  auto prog_repeat = program::Repeat( all_reduce_kernels_repeat[loop], prog);

  Engine engine(graph, prog_repeat);
  // Load the engine on device first
  engine.load(device);
  // Execute on IPUs
  
  gettimeofday(&start, NULL);
  engine.run(0);
  gettimeofday(&end, NULL);

  double time_taken;

    //time_taken must be in seconds, converted 
    // time_taken = (end.tv_sec - start.tv_sec);
  //  time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;

    time_taken = time_taken / all_reduce_kernels_repeat[loop];

    float amount_bytes = numElems * sizeof(type);
    float bw = amount_bytes / time_taken / 1000000000;

    cout << "Time taken by program is : " << fixed
         << time_taken << setprecision(6);
    cout << " sec " << amount_bytes << " bytes " << bw << " GB/s" << endl;

  }

  return 0;
}
