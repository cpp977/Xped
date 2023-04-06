#ifndef XPED_FMT_HELPERS_HPP_
#define XPED_FMT_HELPERS_HPP_

#include "fmt/core.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

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
        using handler_type = fmt::detail::dynamic_specs_handler<format_parse_context>;
        auto type = fmt::detail::type_constant<T, Char>::value;
        fmt::detail::specs_checker<handler_type> handler(handler_type(specs_, ctx), type);
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
        parse_format_specs(ctx.begin(), ctx.end(), handler);
        // todo: fixup alignment
        return base::parse(ctx);
    }
    template <typename FormatCtx>
    auto format(const std::complex<T>& x, FormatCtx& ctx) const -> decltype(ctx.out())
    {
        format_to(ctx.out(), "(");
        if(style_ == style::pair) {
            base::format(x.real(), ctx);
            format_to(ctx.out(), ",");
            base::format(x.imag(), ctx);
            return format_to(ctx.out(), ")");
        }
        if(x.real() || !x.imag()) base::format(x.real(), ctx);
        if(x.imag()) {
            if(x.real() && x.imag() >= 0 && specs_.sign != sign::plus) format_to(ctx.out(), "+");
            base::format(x.imag(), ctx);
            if(style_ == style::star)
                format_to(ctx.out(), "*i");
            else
                format_to(ctx.out(), "i");
            if(std::is_same<typename std::decay<T>::type, float>::value) format_to(ctx.out(), "f");
            if(std::is_same<typename std::decay<T>::type, long double>::value) format_to(ctx.out(), "l");
        }
        return format_to(ctx.out(), ")");
    }
};

#endif