/// \cond
#include "tabulate/row.hpp"
#include "tabulate/table.hpp"
// #include "tabulate/tabulate.hpp"
/// \endcond

#include "Xped/Core/Basis.hpp"

namespace Xped {

Basis Basis::add(const Basis& other) const
{
    Basis out(dim() + other.dim());
    return out;
}

Basis Basis::combine(const Basis& other) const
{
    Basis out(this->dim() * other.dim());
    out.history.dim1 = this->dim();
    out.history.dim2 = other.dim();

    return out;
}

auto Basis::print() const
{
    tabulate::Table t;
    using Row_t = const std::vector<std::variant<std::string, const char*, tabulate::Table>>;
    // TextTable t('-', '|', '+');
    t.add_row(Row_t({"num"}));
    // t.add("num");
    // t.endOfRow();
    for(std::size_t i = 0; i < dim_; i++) {
        std::stringstream ss;
        ss << i;
        t.add_row(Row_t({ss.str()}));
        // t.add(ss.str());
        // t.endOfRow();
    }
    t.format().font_style({tabulate::FontStyle::bold}).border_top("_").border_bottom("_").border_left("|").border_right("|").corner("+");
    t[0].format()
        .padding_top(1)
        .padding_bottom(1)
        .font_align(tabulate::FontAlign::center)
        .font_style({tabulate::FontStyle::underline})
        .font_background_color(tabulate::Color::red);
    return t;
}

auto Basis::printHistory() const
{
    tabulate::Table t;
    using Row_t = const std::vector<std::variant<std::string, const char*, tabulate::Table>>;
    t.add_row(Row_t({"num", "source"}));
    for(std::size_t i = 0; i < dim_; i++) {
        std::stringstream ss, tt;
        ss << i;
        tt << "<-- " << history.source(i)[0] << "," << history.source(i)[1];
        t.add_row(Row_t({ss.str(), tt.str()}));
    }
    t.format().font_style({tabulate::FontStyle::bold}).border_top("_").border_bottom("_").border_left("|").border_right("|").corner("+");
    t[0].format()
        .padding_top(1)
        .padding_bottom(1)
        .font_align(tabulate::FontAlign::center)
        .font_style({tabulate::FontStyle::underline})
        .font_background_color(tabulate::Color::red);
    t[0][1].format().font_background_color(tabulate::Color::blue).font_color(tabulate::Color::white);
    return t;
}

std::ostream& operator<<(std::ostream& os, const Basis& basis)
{
    os << basis.print();
    return os;
}

} // namespace Xped

#if __has_include("Basis.gen.cpp")
#    include "Basis.gen.cpp"
#endif
