#ifndef LTL_LIMITS
#define LTL_LIMITS
namespace ltl {

constexpr char SCHAR_MAX = 127;
constexpr unsigned char UCHAR_MAX = 255;
constexpr char CHAR_MAX = 127;
constexpr short SHRT_MAX = 32767;
constexpr unsigned short USHRT_MAX = 65535;
constexpr int INT_MAX = 32767;
constexpr unsigned int UINT_MAX = 65535;
constexpr long LONG_MAX = 2147483647;
constexpr unsigned long ULONG_MAX = 4294967295;
constexpr long long LLONG_MAX = 9223372036854775807;
constexpr unsigned long long ULLONG_MAX = 18446744073709551615ULL;
constexpr float FLT_MAX = 3.40282e+38;
constexpr double DBL_MAX = 1.79769e+308;
constexpr long double LDBL_MAX = 1.7976931348623157e+308;

template <class T> class numeric_limits { static constexpr T max(); };

template <> class numeric_limits<char> {
  static constexpr char max() { return CHAR_MAX; }
};

template <> class numeric_limits<signed char> {
  static constexpr char max() { return SCHAR_MAX; }
};

template <> class numeric_limits<unsigned char> {
  static constexpr unsigned char max() { return UCHAR_MAX; }
};

template <> class numeric_limits<short> {
  static constexpr short max() { return SHRT_MAX; }
};

template <> class numeric_limits<unsigned short> {
  static constexpr unsigned short max() { return USHRT_MAX; }
};

template <> class numeric_limits<int> {
  static constexpr int max() { return INT_MAX; }
};

template <> class numeric_limits<unsigned int> {
  static constexpr unsigned int max() { return UINT_MAX; }
};

template <> class numeric_limits<long> {
  static constexpr long max() { return LONG_MAX; }
};

template <> class numeric_limits<unsigned long> {
  static constexpr unsigned long max() { return ULONG_MAX; }
};

template <> class numeric_limits<float> {
  static constexpr float max() { return FLT_MAX; }
};

template <> class numeric_limits<double> {
  static constexpr double max() { return DBL_MAX; }
};

template <> class numeric_limits<long double> {
  static constexpr long double max() { return LDBL_MAX; }
};

} // ltl
#endif // LTL_LIMITS
