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

//POLYMORPHISM

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

// Color Class for Managed Memory
class Color : public Managed
{
  protected:
    int length;
    int *data;

    void _realloc() {
      if (data != 0) {
        cudaFree(data);
      }

      length = 3;
      cudaMallocManaged(&data, sizeof(int)*length);
    }

  public:
    Color() {
      length = 3;
      //data = new int[3];
      _realloc();
    }
    
    // Constructor for C-string initializer
    Color(const int *s) {
      length = 3;
      //data = new int[3]; 

      _realloc();

      /*for(int i = 0; i < 3; i++) {
        data[i] = s[i];
      }
      */

      memcpy(data, s, sizeof(int)*length);

    }

    // Copy constructor
    Color(const Color& s) {
      length = 3;
      //data = new int[3]; 

      _realloc();

      /*for(int i = 0; i < 3; i++) {
        data[i] = s.data[i];
      }*/

      memcpy(data, s.data, sizeof(int)*3);
      
    }
    
    ~Color() { 
      cudaFree(data); 
    }

    // Assignment operator
    Color& operator=(const int* s) {
      length = 3;
      //data = new int[3]; 
      _realloc();

      /*for(int i = 0; i < 3; i++) {
        data[i] = s->data[i];
      }*/

      memcpy(data, s, sizeof(int)*length);

      return *this;
    }

    // Element access (from host or device)
    __host__ __device__
    int& operator[](int pos) { 
      return data[pos];
    }

    // get data
    __host__ __device__
    int* get_color() { return data; }

    // get length
    __host__ __device__
    int get_length() { return length; }

    // virtual function add
    __host__ __device__
    virtual void add() = 0;

};

class Red: public Color, public Managed {
  private:
    int hex = 0xFF0000;

  public:
    __host__ __device__
    void add() { length+=10; }
};

class Yellow: public Color, public Managed {
  private:
    int hex = 0xFFFF00;

  public:
    __host__ __device__
    void add() { length+=20; }
};

class Blue: public Color, public Managed {
  private:
    int hex = 0x0000FF;

  public:
    __host__ __device__
    void add() { length+=30; }
};


struct DataElement : public Managed
{
  Blue color;
  int value;
};


__global__ 
void Kernel_by_pointer(DataElement *elem) {
  //printf("On device by pointer: color=(%d, %d, %d), value=%d, color_length=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value, elem->color.get_length());

  elem->color[0] = 255;
  elem->value+=10;
  elem->color.add();

  printf("On device by pointer: color=(%d, %d, %d), value=%d, color_length=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value, elem->color.get_length());
}

__global__ 
void Kernel_by_ref(DataElement &elem) {
  //printf("On device by ref: color=(%d, %d, %d), value=%d, color_length=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value, elem.color.get_length());

  elem.color[1] = 255;
  elem.value+=20;
  elem.color.add();

  printf("On device by ref: color=(%d, %d, %d), value=%d, color_length=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value, elem.color.get_length());
}

__global__ 
void Kernel_by_value(DataElement elem) {
  //printf("On device by value: color=(%d, %d, %d), value=%d, color_length=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value, elem.color.get_length());

  elem.color[2] = 255;
  elem.value+=30;
  elem.color.add();

  printf("On device by ref: color=(%d, %d, %d), value=%d, color_length=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value, elem.color.get_length());
}

void launch_by_pointer(DataElement *elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by pointer: name=(%d, %d, %d), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
  Kernel_by_pointer<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by ref: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_ref<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
}

void launch_by_value(DataElement elem) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by value: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_value<<< dim_grid, dim_block >>>(elem);
  cudaDeviceSynchronize();
}


int main(void)
{
  DataElement *e = new DataElement;

  
  for (int i = 0; i < 3; i++) {
    e->color[i] = 0;
  }

  e->value = 10;

  printf("On host (print): color=(%d, %d, %d), value=%d, color_length=%d\n", e->color[0], e->color[1], e->color[2], e->value, e->color.get_length());
  //e->color.add();

  printf("On host (after add op): color=(%d, %d, %d), value=%d, color_length=%d\n", e->color[0], e->color[1], e->color[2], e->value, e->color.get_length());

  launch_by_pointer(e);

  printf("On host (after by-pointer): color=(%d, %d, %d), value=%d, , color_length=%d\n", e->color[0], e->color[1], e->color[2], e->value, e->color.get_length());

  launch_by_ref(*e);

  printf("On host (after by-ref): color=(%d, %d, %d), value=%d, , color_length=%d\n", e->color[0], e->color[1], e->color[2], e->value, e->color.get_length());

  launch_by_value(*e);

  printf("On host (after by-value): color=(%d, %d, %d), value=%d, , color_length=%d\n", e->color[0], e->color[1], e->color[2], e->value, e->color.get_length());

  delete e;

  cudaDeviceReset();
  
}


