template<std::size_t Rank>
std::string printTree(const std::array<std::string,Rank>& uncoupled, const std::array<std::string,util::inter_dim(Rank)>& intermediates, const std::string& coupled, const std::array<std::string,util::mult_dim(Rank)>& multiplicities, const std::array<bool,Rank> IS_DUAL) {
        return "";
}

template<>
std::string printTree<4>(const std::array<std::string,4>& uncoupled, const std::array<std::string,2>& intermediates, const std::string& coupled, const std::array<std::string,3>& multiplicities, const std::array<bool,4> IS_DUAL)
{
        std::stringstream ss;
        if (IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2] and !IS_DUAL[3])      {ss << "  ☐" << endl;}
        else if (!IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and !IS_DUAL[3]) {ss << "           ☐" << endl;}
        else if (!IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3]) {ss << "                    ☐" << endl;}
        else if (!IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3]) {ss << "                             ☐" << endl;}
        
        else if (IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and !IS_DUAL[3])  {ss << "  ☐        ☐" << endl;}
        else if (IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3])  {ss << "  ☐                 ☐" << endl;}
        else if (IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3])  {ss << "  ☐                          ☐" << endl;}
        else if (!IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3])  {ss << "           ☐        ☐" << endl;}
        else if (!IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3])  {ss << "           ☐                 ☐" << endl;}
        else if (!IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3])  {ss << "                    ☐        ☐" << endl;}
        
        if (!IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3])        {ss << "           ☐        ☐        ☐" << endl;}
        else if (IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3])   {ss << "  ☐                 ☐        ☐" << endl;}
        else if (IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2] and IS_DUAL[3])   {ss << "  ☐        ☐                 ☐" << endl;}
        else if (IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and !IS_DUAL[3])   {ss << "  ☐        ☐        ☐" << endl;}
                
        else if (IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2] and IS_DUAL[3])  {ss <<   "  ☐        ☐        ☐        ☐" << endl;}

        ss << uncoupled[0] << "    " << uncoupled[1] << "     " << uncoupled[2] << "      " << uncoupled[3] << endl;
        ss << "   \\     /        /        /\n";
        ss << "    \\   /        /        /\n";
        ss << "     " << multiplicities[0] << "         /        /\n";
        ss << "       \\       /        /\n";
        ss << "        \\" << intermediates[0] << " /        /\n";
        ss << "         \\   /        /\n";
        ss << "          " << multiplicities[1] << "         /\n";
        ss << "            \\       /\n";
        ss << "             \\" << intermediates[1] << " /\n";
        ss << "              \\   /\n";
        ss << "               " << multiplicities[2] << "\n";
        ss << "                |\n";
        ss << "                |\n";
        ss << "               " << coupled << "\n";
        return ss.str();
};

template<>
std::string printTree<3>(const std::array<std::string,3>& uncoupled, const std::array<std::string,1>& intermediates, const std::string& coupled, const std::array<std::string,2>& multiplicities, const std::array<bool,3> IS_DUAL)
{
        std::stringstream ss;
        if (IS_DUAL[0] and !IS_DUAL[1] and !IS_DUAL[2])      {ss << "  ☐" << endl;}
        else if (!IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2]) {ss << "           ☐" << endl;}
        else if (!IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2]) {ss << "                    ☐" << endl;}
        else if (IS_DUAL[0] and IS_DUAL[1] and !IS_DUAL[2]) {ss <<  "  ☐        ☐" << endl;}
        else if (IS_DUAL[0] and !IS_DUAL[1] and IS_DUAL[2]) {ss <<  "  ☐                 ☐" << endl;}
        else if (!IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2]) {ss <<  "           ☐        ☐" << endl;}
        else if (IS_DUAL[0] and IS_DUAL[1] and IS_DUAL[2]) {ss <<   "  ☐        ☐        ☐" << endl;}
        ss << uncoupled[0] << "    " << uncoupled[1] << "      " << uncoupled[2] << endl;
        ss << "   \\     /        /\n";
        ss << "    \\   /        /\n";
        ss << "     " << multiplicities[0] << "         /\n";
        ss << "       \\       /\n";
        ss << "        \\" << intermediates[0] << " /\n";
        ss << "         \\   /\n";
        ss << "          " << multiplicities[1] << "\n";
        ss << "           |\n";
        ss << "           |\n";
        ss << "          " << coupled << "\n";
        return ss.str();        
};

template<>
std::string printTree<2>(const std::array<std::string,2>& uncoupled, const std::array<std::string,0>&, const std::string& coupled, const std::array<std::string,1>& multiplicities, const std::array<bool,2> IS_DUAL)
{
        std::stringstream ss;
        if (IS_DUAL[0] and !IS_DUAL[1])      {ss << "  ☐" << endl;}
        else if (IS_DUAL[1] and !IS_DUAL[0]) {ss << "           ☐" << endl;}
        else if (IS_DUAL[0] and IS_DUAL[1])  {ss << "  ☐        ☐" << endl;}
        ss << uncoupled[0] << "    " << uncoupled[1] << endl;;
        ss << "  \\     /\n";
        ss << "   \\   /\n";
        ss << "    " <<  multiplicities[0] << endl;
        ss << "     |\n";
        ss << "     |\n";
        ss << "    " << coupled << endl;
        return ss.str();
};

template<>
std::string printTree<1>(const std::array<std::string,1>& uncoupled, const std::array<std::string,0>&, const std::string& coupled, const std::array<std::string,0>&, const std::array<bool,1> IS_DUAL)
{
        assert(uncoupled[0] == coupled);
        std::stringstream ss;
        if (IS_DUAL[0]) {ss << " ☐" << endl;}
        ss << uncoupled[0] << "\n";
        ss << " |\n";
        ss << " |\n";
        ss << " |\n";
        ss << " |\n";
        ss << coupled << "\n";
        return ss.str();
};

template<>
std::string printTree<0>(const std::array<std::string,0>&, const std::array<std::string,0>&, const std::string&, const std::array<std::string,0>&, const std::array<bool,0>)
{
        std::stringstream ss;
        ss << "0\n";
        ss << "|\n";
        ss << "|\n";
        ss << "|\n";
        ss << "|\n";
        ss << "0\n";
        return ss.str();
};

