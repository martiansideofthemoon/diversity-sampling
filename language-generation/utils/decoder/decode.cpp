#include "beam.hpp"
#include <ctime>


void build_vocab(string vocab_file, map<string, int> &vocab, vector<string> &rev_vocab) {
    ifstream infile(vocab_file.c_str());
    string line;
    string token = "";
    int id = -1;
    while (getline(infile, line)) {
        istringstream iss(line);
        if (line.empty()) {
            continue;
        }
        id++;
        iss >> token;
        vocab.insert(pair<string, int>(token, id));
        rev_vocab.push_back(token);
    }
    infile.close();
}


void load_ctc(string data_file, list<SequenceElement> &sentences, int vocab_size) {
    ifstream infile(data_file.c_str());
    string line;
    string token = "";
    int id = 0;
    while (getline(infile, line)) {
        if (line.empty()) {
            continue;
        }
        id++;
        SequenceElement* sq = new SequenceElement(line, vocab_size);
        sentences.push_back(*sq);
    }
    infile.close();
}


int main(int argc, char *argv[]) {
    Trie* tr = new Trie();
    map<string, int> vocab;
    vector<string> rev_vocab;
    build_vocab(argv[1], vocab, rev_vocab);
    cout << "Vocab loaded." << endl;
    tr->load_arpa(argv[2], vocab);
    list<SequenceElement> sentences;
    load_ctc(argv[3], sentences, vocab.size() - 3 + 1);
    cout << "CTC probs loaded." << endl;
    std::clock_t  start;
    start = std::clock();
    if (strcmp(argv[7], "greedy") == 0) {
        cout << "Using greedy decoder." << endl;
        greedy_decoder(sentences, rev_vocab, vocab);
    } else {
        cout << "Using beam search decoder." << endl;
        bool sos_choice = false;
        if (strcmp(argv[8], "true") == 0) {
            sos_choice = true;
        }
        beam_decoder(sentences, *tr, lexical_cast<float>(argv[4]),
                     lexical_cast<float>(argv[5]), lexical_cast<int>(argv[6]),
                     sos_choice, rev_vocab, vocab);
    }
    double time_taken = (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
    std::ofstream out(argv[9]);
    for (list<SequenceElement>::iterator it = sentences.begin(); it != sentences.end(); it++) {
        out << it->sent_id << "\t" << it->decoded << endl;
    }
    out.close();
    cout << "Time: " << time_taken << " ms" << std::endl;
}
