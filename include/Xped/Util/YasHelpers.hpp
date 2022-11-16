#ifndef XPED_YAS_HELPERS_HPP
#define XPED_YAS_HELPERS_HPP

#include <filesystem>

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

namespace yas {
namespace detail {

/***************************************************************************/

template <std::size_t F>
struct serializer<type_prop::not_a_fundamental, ser_case::use_internal_serializer, F, std::filesystem::path>
{
    template <typename Archive>
    static Archive& save(Archive& ar, const std::filesystem::path& p)
    {
        __YAS_CONSTEXPR_IF(F & yas::json)
        {
            if(p.string().empty()) {
                ar.write("null", 4);
            } else {
                ar.write("\"", 1);
                save_string(ar, p.string().data(), p.string().size());
                ar.write("\"", 1);
            }
        }
        else
        {
            ar.write_seq_size(p.string().length());
            ar.write(p.string().data(), p.string().length());
        }
        return ar;
    }

    template <typename Archive>
    static Archive& load(Archive& ar, std::filesystem::path& p)
    {
        std::string tmp;

        __YAS_CONSTEXPR_IF(F & yas::json)
        {
            char ch = ar.getch();
            if(ch == '\"') {
                load_string(tmp, ar);
                __YAS_THROW_IF_WRONG_JSON_CHARS(ar, "\"");
            } else if(ch == 'n') {
                ar.ungetch(ch);
                __YAS_THROW_IF_WRONG_JSON_CHARS(ar, "null");
                tmp.clear();
            } else if(is_valid_for_int_and_double(ar, ch)) {
                tmp += ch;
                for(ch = ar.peekch(); is_valid_for_int_and_double(ar, ch); ch = ar.peekch()) { tmp += ar.getch(); }
            } else {
                __YAS_THROW_IF_WRONG_JSON_CHARS(ar, "unreachable");
            }
        }
        else
        {
            const auto size = ar.read_seq_size();
            tmp.resize(size);
            ar.read(__YAS_CCAST(char*, tmp.data()), size);
        }
        p = tmp;
        return ar;
    }
};

/***************************************************************************/

} // namespace detail
} // namespace yas

#endif
