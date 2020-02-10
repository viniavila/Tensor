#include "Tensor.h"
#include <cstring>
#include <utility>
#include <iostream>
#include <random>

#ifdef ENABLED_MULTITHREADING
#include <ThreadPool.h>

namespace ItemOperations {
    void multiplication(const Tensor* a, const Tensor *b, uint32_t i, uint32_t j, Tensor *r) {
        double sum = 0.0;
        for (uint32_t k = 0; k < a->columns(); ++k)
            sum += a->value(i, k) * b->value(k, j);
        r->setValue(i, j, sum);
    }

    void sum(uint32_t i, const Tensor *r, const Tensor *o) {
        double *pR = r->data()+(i*r->columns());
        double *pO = o->data()+(i*o->columns());
        for (uint32_t j = 0; j < r->columns(); ++j, pR++, pO++)
            *pR += *pO;
    }

    void subtract(uint32_t i, const Tensor *r, const Tensor *o) {
        double *pR = r->data()+(i*r->columns());
        double *pO = o->data()+(i*o->columns());
        for (uint32_t j = 0; j < r->columns(); ++j, pR++, pO++)
            *pR -= *pO;
    }

    void element_product(uint32_t i, const Tensor *r, const Tensor *o) {
        double *pR = r->data()+(i*r->columns());
        double *pO = o->data()+(i*o->columns());
        for (uint32_t j = 0; j < r->columns(); ++j, pR++, pO++)
            *pR *= *pO;
    }

    void scalar_product(uint32_t i, const Tensor *r, double b) {
        double *pR = r->data()+(i*r->columns());
        for (uint32_t j = 0; j < r->columns(); ++j, pR++)
            *pR *= b;
    }

    void map(uint32_t i, const Tensor *in, const Tensor *out, const std::function<double(double)>& f) {
        double *pI = in->data()+(i*in->columns());
        double *pO = out->data()+(i*out->columns());
        for (uint32_t j = 0; j < out->columns(); ++j, pI++, pO++)
            *pO = f(*pI);
    }

    void reduce_row(uint32_t i, const Tensor *in, double *sum, const std::function<double(double)>& f) {
        double *pI = in->data()+(i*in->columns());
        *sum = 0.0;
        for (uint32_t j = 0; j < in->columns(); ++j, pI++)
            *sum += f(*pI);
    }

    void transpose_row(uint32_t i, const Tensor *in, Tensor *out) {
        double *pI = in->data()+(i*in->columns());
        double *pO = out->data()+i;
        for (uint32_t j = 0; j < in->columns(); ++j, pI++, pO+=in->rows())
            *pO = *pI;
    }
}

#endif

struct TensorData {
    uint32_t rows;
    uint32_t columns;
    double * data;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

    inline TensorData() : rows(0), columns(0), data(nullptr) { }

    TensorData(TensorData* c) : rows(c->rows), columns(c->columns) {
        data = new double[rows*columns];
        std::memcpy(data, c->data, sizeof(double)*rows*columns);
        distribution = c->distribution;
    }

    TensorData(uint32_t r, uint32_t c) : rows(r), columns(c) {
        data = new double[r*c];
        std::fill(data, data+(r*c), 0.0);
    }

    ~TensorData() {
        if (data != nullptr) delete[] data;
    }

    uint32_t indexOf(uint32_t i, uint32_t j) const {
        return i*columns+j;
    }   
};

Tensor::Tensor() : d(new TensorData) {
    std::cout << "Default Constructor - Create an Empty and invalid Tensor" << std::endl;
}

Tensor::Tensor(uint32_t r, uint32_t c, double min, double max) : d(new TensorData(r, c)) {
    std::cout << "Parameter Constructor - Create a valid r x c Tensor filled with random numbers between min and max" << std::endl;
    d->distribution = std::uniform_real_distribution<double>(min, max);
    double *p = d->data;
    for (uint32_t i = 0; i < r*c; ++i, p++)
        *p = d->distribution(d->generator);
}

Tensor::Tensor(uint32_t r, uint32_t c, const std::initializer_list<double>& data) :
    d(new TensorData(r, c))
{
    std::cout << "Parameter Constructor - Create a valid r x c Tensor filled with values in data (and completed with zeroes if data.size() < r*c)" << std::endl;
    if (data.size()) {
        if (r*c > data.size()) {
            uint32_t i = 0;
            for (auto iter = data.begin(); iter != data.end(); ++iter) {
                d->data[i] = *iter;
                i++;
            }
        }
        else {
            auto iter = data.begin();
            for (uint32_t i = 0; i < r*c; ++i) {
                d->data[i] = *iter;
                iter++;
            }
        }
    }
}

Tensor::Tensor(const Tensor& c) : d(new TensorData(c.d)) {
    std::cout << "Copy Constructor - Duplicates the Tensor passed as argument" << std::endl;
}

Tensor& Tensor::operator=(const Tensor& c) {
    if (d != nullptr) delete d;
    d = new TensorData(c.d);
    std::cout << "Copy Assignment" << std::endl;
    return *this;
}

Tensor::Tensor(Tensor&& m) : d(std::move(m.d)) {
    std::cout << "Move Constructor - Creates a Tensor and takes the ownership of the temporary Tensor data" << std::endl;
}

Tensor& Tensor::operator=(Tensor&& m) {
    if (d != nullptr) delete d;
    d = std::move(m.d);
    std::cout << "Move Assignment" << std::endl;
    return *this;
}

uint32_t Tensor::rows() const {
    return d->rows;
}

uint32_t Tensor::columns() const {
    return d->columns;
}

double Tensor::value(uint32_t i, uint32_t j) const {
    return d->data[d->indexOf(i, j)];
}

double& Tensor::operator()(uint32_t i, uint32_t j) {
    return d->data[d->indexOf(i, j)];
}

double& Tensor::operator()(uint32_t i, uint32_t j) const {
    return d->data[d->indexOf(i, j)];
}

void Tensor::setValue(uint32_t i, uint32_t j, double value) {
    d->data[d->indexOf(i, j)] = value;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    // Check this size and other size
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i<d->rows; ++i)
        pool.push_job(std::bind(ItemOperations::sum, i, this, &other));
    pool.wait();
#else
    for (uint32_t i = 0; i<d->rows*d->columns; ++i)
        d->data[i] += other.d->data[i];
#endif
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    // Check this size and other size
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i<d->rows; ++i)
        pool.push_job(std::bind(ItemOperations::subtract, i, this, &other));
    pool.wait();
#else
    for (uint32_t i = 0; i<d->rows*d->columns; ++i)
        d->data[i] -= other.d->data[i];
#endif
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    // Check this size and other size
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i<d->rows; ++i)
        pool.push_job(std::bind(ItemOperations::element_product, i, this, &other));
    pool.wait();
#else
    for (uint32_t i = 0; i<d->rows*d->columns; ++i)
        d->data[i] *= other.d->data[i];
#endif
    return *this;
}

Tensor& Tensor::operator*=(double b) {
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i<d->rows; ++i)
        pool.push_job(std::bind(ItemOperations::scalar_product, i, this, b));
    pool.wait();
#else
    for (uint32_t i = 0; i<d->rows*d->columns; ++i)
        d->data[i] *= b;
#endif
    return *this;
}

Tensor Tensor::map(const std::function<double(double)>& f) const {
    Tensor out(*this);
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i<d->rows; ++i)
        pool.push_job(std::bind(ItemOperations::map, i, this, &out, f));
    pool.wait();
#else
    for (uint32_t i = 0; i<d->rows*d->columns; ++i)
        out.d->data[i] = f(d->data[i]);
#endif
    return out;
}

double Tensor::reduce(const std::function<double(double)>& f) const {
    double out = 0.0;
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    double o_row[d->rows];
    for (uint32_t i = 0; i<d->rows; ++i) {
        o_row[i] = 0.0;
        pool.push_job(std::bind(ItemOperations::reduce_row, i, this, &o_row[i], f));
    }
    pool.wait();
    double *pO = o_row;
    for (uint32_t i = 0; i<d->rows; ++i, pO++)
        out += *pO;
#else
    for (uint32_t i = 0; i<d->rows*d->columns; ++i)
        out += f(d->data[i]);
#endif
    return out;
}

Tensor Tensor::transpose() const {
    Tensor out(d->columns, d->rows);
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i<d->rows; ++i)
        pool.push_job(std::bind(ItemOperations::transpose_row, i, this, &out));
    pool.wait();
#else
    double *p = d->data;
    for (uint32_t i = 0; i<d->rows; ++i)
        for (uint32_t j = 0; j<d->columns; ++j, p++)
            out(j,i) = *p;
#endif
    return out;
}

Tensor Tensor::getRow(uint32_t row, bool transposed) const {
    Tensor out(1,d->columns);
    double *pI = d->data+(row*d->columns);
    double *pO = out.data();
    for (uint32_t j = 0; j < d->columns; ++j, pI++, pO++)
        *pO = *pI;
    if (transposed) {
        out.d->rows = d->columns;
        out.d->columns = 1;
    }
    return out;
}

Tensor Tensor::getColumn(uint32_t column, bool transposed) const {
    Tensor out(d->rows,1);
    double *pI = d->data+column;
    double *pO = out.data();
    for (uint32_t j = 0; j < d->rows; ++j, pI+=d->columns, pO++)
        *pO = *pI;
    if (transposed) {
        out.d->columns = out.d->rows;
        out.d->rows = 1;
    }
    return out;
}

#ifdef ENABLED_MULTITHREADING
double* Tensor::data() const {
    return d->data;
}
#endif

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "[";
    for (uint32_t i = 0; i < t.rows(); ++i) {
        os << "[";
        for (uint32_t j = 0; j<t.columns(); ++j)
            os << t.value(i,j) << ",";
        os << "\b],";
    }
    os << "\b]";
    return os;
}

Tensor operator+(Tensor a, const Tensor& b) {
    a+=b;
    return a;
}

Tensor operator-(Tensor a, const Tensor& b) {
    a-=b;
    return a;
}

Tensor operator*(Tensor a, const Tensor& b) {
    a*=b;
    return a;
}

Tensor operator*(Tensor a, double b) {
    std::cout << "called" << std::endl;
    a*=b;
    return a;
}

Tensor operator*(double b, Tensor a) {
    a*=b;
    return a;
}

Tensor operator%(const Tensor& a, const Tensor& b) {
    // Check dimensions of a and b
    Tensor s(a.rows(), b.columns());
#ifdef ENABLED_MULTITHREADING
    ThreadPool& pool = ThreadPool::getInstance();
    for (uint32_t i = 0; i < s.rows(); ++i)
        for (uint32_t j = 0; j < s.columns(); ++j)
            pool.push_job(std::bind(ItemOperations::multiplication, &a, &b, i, j, &s));
    pool.wait();
#else
    for (uint32_t i = 0; i < s.rows(); ++i) {
        for (uint32_t j = 0; j < s.columns(); ++j) {
            double acc = 0.0;
            for (uint32_t k = 0; k < a.columns(); ++k)
                acc += a(i,k) * b(k,j);
            s(i,j) = acc;
        }
    }
#endif
    return s;
}
