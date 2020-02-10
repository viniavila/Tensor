#ifndef TENSOR_H
#define TENSOR_H

#include <cinttypes>
#include <iostream>
#include <functional>

struct TensorData;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(uint32_t r, uint32_t c, double min, double max);
    Tensor(uint32_t r, uint32_t c, const std::initializer_list<double>& data = { });
    Tensor(const Tensor& c);
    Tensor(Tensor&& m);

    // Assignment Operators
    Tensor& operator=(const Tensor& c);
    Tensor& operator=(Tensor&& m);

    // Tensor Characteristics and Value
    uint32_t rows() const;
    uint32_t columns() const;
    double value(uint32_t i, uint32_t j) const;
    double& operator()(uint32_t i, uint32_t j) const;

    // Value Assignment (By Indexes)
    double& operator()(uint32_t i, uint32_t j);
    void setValue(uint32_t i, uint32_t j, double value);

    // Tensor Unary Operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator*=(double b);

    // Tensor Basic Methods
    Tensor map(const std::function<double(double)>& f) const;
    double reduce(const std::function<double(double)>& f) const;
    Tensor transpose() const;

    // Extract Methods
    Tensor getRow(uint32_t row, bool transposed = false) const;
    Tensor getColumn(uint32_t column, bool transposed = false) const;

#ifdef ENABLED_MULTITHREADING
    double* data() const;
#endif

private:
    TensorData * d;
};

// Tensor output streaming
std::ostream& operator<<(std::ostream& os, const Tensor& t);

// Tensor non-member operations
Tensor operator+(Tensor a, const Tensor& b);
Tensor operator-(Tensor a, const Tensor& b);
Tensor operator*(Tensor a, const Tensor& b);
Tensor operator*(Tensor a, double b);
Tensor operator*(double b, Tensor a);
Tensor operator%(const Tensor& a, const Tensor& b);
#endif // TENSOR_H
