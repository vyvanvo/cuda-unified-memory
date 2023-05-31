#include <cmath>
#include <iostream>
#include <cassert>
#include "managed.h"

static const double pi = 4 * atan(1.0);

template<class T, int n> struct vec;
template<class T, int n> T dot(const vec<T,n>& u,const vec<T,n>& v);

template<class T, int n>
struct vec: public Managed
{
    T x[n];

    //initializing vec
    vec()
    {make_zero();}

    explicit vec(const T& a)
    {assert(n == 1);x[0]=a;}

    vec(const T& a, const T& b)
    {assert(n == 2);x[0]=a;x[1]=b;}

    vec(const T& a, const T& b, const T& c)
    {assert(n == 3);x[0]=a;x[1]=b;x[2]=c;}

    template<class U>
    explicit vec(const vec<U,n>& v)
    {for(int i = 0; i < n; i++) x[i] = (T)v.x[i];}

    //functions
    __host__ __device__
    void make_zero()
    {fill(0);}

    void fill(T value)
    {for(int i = 0; i < n; i++) x[i] = value;}

    vec& operator += (const vec& v)
    {for(int i = 0; i < n; i++) x[i] += v.x[i]; return *this;}

    vec& operator -= (const vec& v)
    {for(int i = 0; i < n; i++) x[i] -= v.x[i]; return *this;}

    vec& operator *= (const vec& v)
    {for(int i = 0; i < n; i++) x[i] *= v.x[i]; return *this;}

    vec& operator /= (const vec& v)
    {for(int i = 0; i < n; i++) x[i] /= v.x[i]; return *this;}

    __host__ __device__
    vec& operator += (const T& c)
    {for(int i = 0; i < n; i++) x[i] += c; return *this;}

    vec& operator -= (const T& c)
    {for(int i = 0; i < n; i++) x[i] -= c; return *this;}

    vec& operator *= (const T& c)
    {for(int i = 0; i < n; i++) x[i] *= c; return *this;}

    vec& operator /= (const T& c)
    {for(int i = 0; i < n; i++) x[i] /= c; return *this;}

    vec operator + () const
    {return *this;}

    vec operator - () const
    {vec r; for(int i = 0; i < n; i++) r[i] = -x[i]; return r;}

    vec operator + (const vec& v) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] + v.x[i]; return r;}

    vec operator - (const vec& v) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] - v.x[i]; return r;}

    vec operator * (const vec& v) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] * v.x[i]; return r;}

    vec operator / (const vec& v) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] / v.x[i]; return r;}

    vec operator + (const T& c) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] + c; return r;}

    vec operator - (const T& c) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] - c; return r;}

    vec operator * (const T& c) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] * c; return r;}

    vec operator / (const T& c) const
    {vec r; for(int i = 0; i < n; i++) r[i] = x[i] / c; return r;}

    const T& operator[] (int i) const
    {return x[i];}

    T& operator[] (int i)
    {return x[i];}

    T magnitude_squared() const
    {return dot(*this, *this);}

    T magnitude() const
    {return sqrt(magnitude_squared());}

    // Be careful to handle the zero vector gracefully
    __host__ __device__
    vec normalized() const
    {T mag = magnitude(); if(mag) return *this / mag; vec r; r[0] = 1; return r;};
};

template <class T, int n>
vec<T,n> operator * (const T& c, const vec<T,n>& v)
{return v*c;}

template <class T, int n>
T dot(const vec<T,n> & u, const vec<T,n> & v)
{
    T r  =  0;
    for(int i = 0; i < n; i++) r += u.x[i] * v.x[i];
    return r;
}

template <class T >
vec<T,3> cross(const vec<T,3> & u, const vec<T,3> & v)
{
    return vec<T,3> (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]);
}

template<class T, int d>
vec<T,d> componentwise_max(const vec<T,d>& a, const vec<T,d>& b)
{
    vec<T,d> r;
    for(int i=0; i<d; i++) r[i] = std::max(a[i], b[i]);
    return r;
}

template<class T, int d>
vec<T,d> componentwise_min(const vec<T,d>& a, const vec<T,d>& b)
{
    vec<T,d> r;
    for(int i=0; i<d; i++) r[i] = std::min(a[i], b[i]);
    return r;
}

template <class T, int n>
std::ostream& operator << (std::ostream& out, const vec<T,n> & u)
{
    out << '(';
    for(int i = 0; i < n; i++)
    {
        if(i) out << ' ';
        out << u[i];
    }
    out << ')';
    return out;
}

template <class T, int n>
std::istream& operator >> (std::istream& in, vec<T,n> & u)
{
    for(int i = 0; i < n; i++)
    {
        in >> u[i];
    }
    return in;
}

using std::abs;

template <class T, int n>
vec<T,n> abs(const vec<T,n> & u)
{
    vec<T,n> r;
    for(int i = 0; i < n; i++)
        r[i] = std::abs(u[i]);
    return r;
}

typedef vec<double,2> vec2;
typedef vec<double,3> vec3;
typedef vec<double,4> vec4;
typedef vec<int,2> ivec2;
typedef vec<int,3> ivec3;
typedef vec<int,4> ivec4;

struct DataElement : public Managed
{
  vec3 color;
  int value;
};


__global__ 
void Kernel_by_pointer(DataElement *elem, DataElement *elem2) {
  printf("On device by pointer (before changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
  
  //elem->color[0] = 255;
  elem->value+=10;

  vec3 color2 = {100, 100, 100};
  elem->color += elem2->color;
  elem->color += color2;

  printf("On device by pointer (after changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
}

__global__ 
void Kernel_by_ref(DataElement &elem, DataElement &elem2) {
  printf("On device by ref (before changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);

  //elem.color[1] = 255;
  elem.value+=20;
  elem.color += elem2.color;

  printf("On device by ref (after changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
}

__global__ 
void Kernel_by_value(DataElement elem, DataElement elem2) {
  printf("On device by value (before changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);

  //elem.color[2] = 255;
  elem.value+=30;
  elem.color += elem2.color;

  printf("On device by value (after changes): color=(%.2f, %.2f, %.2f), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
}

void launch_by_pointer(DataElement *elem, DataElement *elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by pointer: name=(%d, %d, %d), value=%d\n", elem->color[0], elem->color[1], elem->color[2], elem->value);
  Kernel_by_pointer<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem, DataElement &elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by ref: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_ref<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}

void launch_by_value(DataElement elem, DataElement elem2) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(1, 1, 1);

  //printf("launch by value: name=(%d, %d, %d), value=%d\n", elem.color[0], elem.color[1], elem.color[2], elem.value);
  Kernel_by_value<<< dim_grid, dim_block >>>(elem, elem2);
  cudaDeviceSynchronize();
}


int main(void)
{
  DataElement *e = new DataElement;
  DataElement *f = new DataElement;
  
  for (int i = 0; i < 3; i++) {
    e->color[i] = 10;
  }

  e->value = 10;

  for (int i = 0; i < 3; i++) {
    f->color[i] = 20;
  }

  f->value = 100;

  printf("On host (print) e: color=(%.2f, %.2f, %.2f), value=%d\n", e->color[0], e->color[1], e->color[2], e->value);
  printf("On host (print) f: color=(%.2f, %.2f, %.2f), value=%d\n", f->color[0], f->color[1], f->color[2], f->value);

  //add
  e->color += f->color;
  printf("On host (after e + f): color=(%.2f, %.2f, %.2f), value=%d\n", e->color[0], e->color[1], e->color[2], e->value);
  
  launch_by_pointer(e, f);

  printf("On host (after by-pointer): color=(%.2f, %.2f, %.2f), value=%d\n", e->color[0], e->color[1], e->color[2], e->value);

  launch_by_ref(*e, *f);

  printf("On host (after by-ref): color=(%.2f, %.2f, %.2f), value=%d\n", e->color[0], e->color[1], e->color[2], e->value);

  launch_by_value(*e, *f);

  printf("On host (after by-value): color=(%.2f, %.2f, %.2f), value=%d\n", e->color[0], e->color[1], e->color[2], e->value);

  delete e;

  cudaDeviceReset();
  
}


