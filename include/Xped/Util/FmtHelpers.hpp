#ifndef XPED_FMT_HELPERS_HPP_
#define XPED_FMT_HELPERS_HPP_

#include "fmt/core.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

// template <typename T, typename Char>
// struct fmt::formatter<std::complex<T>, Char> : ostream_formatter
// {};

template <typename T, typename Char>
struct fmt::formatter<std::complex<T>, Char> : public fmt::formatter<T, Char>
{
    typedef fmt::formatter<T, Char> base;
    enum style
    {
        expr,
        star,
        pair
    } style_ = expr;
    fmt::detail::dynamic_format_specs<Char> specs_;
    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin())
    {
        auto it = ctx.begin();
        if(it != ctx.end()) {
            switch(*it) {
            case '$':
                style_ = style::expr;
                ctx.advance_to(++it);
                break;
            case '*':
                style_ = style::star;
                ctx.advance_to(++it);
                break;
            case ',':
                style_ = style::pair;
                ctx.advance_to(++it);
                break;
            default: break;
            }
        }
        auto type = fmt::detail::type_constant<T, Char>::value;
        parse_format_specs(ctx.begin(), ctx.end(), specs_, ctx, type);
        // todo: fixup alignment
        return base::parse(ctx);
    }
    template <typename FormatCtx>
    auto format(const std::complex<T>& x, FormatCtx& ctx) const -> decltype(ctx.out())
    {
		fmt::format_to(ctx.out(), "(");
        if(style_ == style::pair) {
            base::format(x.real(), ctx);
            fmt::format_to(ctx.out(), ",");
            base::format(x.imag(), ctx);
            return fmt::format_to(ctx.out(), ")");
        }
        if(x.real() || !x.imag()) base::format(x.real(), ctx);
        if(x.imag()) {
            if(x.real() && x.imag() >= 0 && specs_.sign != sign::plus) fmt::format_to(ctx.out(), "+");
            base::format(x.imag(), ctx);
            if(style_ == style::star)
				fmt::format_to(ctx.out(), "*i");
            else
				fmt::format_to(ctx.out(), "i");
            if(std::is_same<typename std::decay<T>::type, float>::value) fmt::format_to(ctx.out(), "f");
            if(std::is_same<typename std::decay<T>::type, long double>::value) fmt::format_to(ctx.out(), "l");
        }
        return fmt::format_to(ctx.out(), ")");
    }
};

#endif
