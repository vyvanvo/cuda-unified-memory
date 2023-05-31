/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <string.h>
#include <stdio.h>

//CLASS DATA ELEM WITH INT

// Managed Base Class -- inherit from this to automatically 
// allocate objects in Unified Memory
class Managed 
{
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

class DataElement : public Managed
{
  public:
    int key;
    int value;
};

__global__ 
void Kernel_by_pointer(DataElement *elem) {
  printf("On device by pointer:       key= %d, value=%d\n", elem->key, elem->value);

  elem->key+=5;
  elem->value+=10;
}

__global__ 
void Kernel_by_ref(DataElement &elem) {
  printf("On device by ref:           name=%d, value=%d\n", elem.key, elem.value);

  elem.key+=5;
  elem.value+=20;
}

__global__ 
void Kernel_by_value(DataElement elem) {
  printf("On device by value:         name=%d, value=%d\n", elem.key, elem.value);

  elem.key+=5;
  elem.value+=30;
}

void launch_by_pointer(DataElement *elem) {
  Kernel_by_pointer<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem) {
  Kernel_by_ref<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_value(DataElement elem) {
  Kernel_by_value<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

int main(void)
{
  DataElement *e = new DataElement;
  
  e->value = 10;
  e->key = 0;
  
  launch_by_pointer(e);

  printf("On host (after by-pointer): name=%d, value=%d\n", e->key, e->value);

  launch_by_ref(*e);

  printf("On host (after by-ref):     name=%d, value=%d\n", e->key, e->value);

  launch_by_value(*e);

  printf("On host (after by-value):   name=%d, value=%d\n", e->key, e->value);

  //delete e;

  cudaDeviceReset();
}


