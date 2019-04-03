import os
import util
import numpy as np

# @profile
def write_highlighted_html(html, out_dir, example_idx):
    html = '''

<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d_highlighted.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d_highlighted.html";
    };

    document.addEventListener("keyup",function(e){
   var key = e.which||e.keyCode;
   switch(key){
      //left arrow
      case 37:
         document.getElementById("btnPrev").click();
      break;
      //right arrow
      case 39:
         document.getElementById("btnNext").click();
      break;
   }
});
</script>

''' % (example_idx - 1, example_idx + 1) + html
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    with open(path, 'w') as f:
        f.write(html)

highlight_colors = ['aqua', 'lime', 'yellow', '#FF7676', '#B9968D', '#D7BDE2', '#D6DBDF', '#F852AF', '#00FF8B', '#FD933A', '#8C8DFF', '#965DFF']
hard_highlight_colors = ['#00BBFF', '#00BB00', '#F4D03F', '#BB5454', '#A16252', '#AF7AC5', '#AEB6BF', '#FF008F', '#0ECA74', '#FF7400', '#6668FF', '#7931FF']
# hard_highlight_colors = ['blue', 'green', 'orange', 'red']

def start_tag(color):
    return "<font color='" + color + "'>"

def start_tag_highlight(color):
    return "<mark style='background-color: " + color + ";'>"

def get_idx_for_source_idx(similar_source_indices, source_idx):
    summ_sent_indices = []
    priorities = []
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        for idx_idx, idx in enumerate(source_indices):
            if source_idx == idx:
                summ_sent_indices.append(source_indices_idx)
                priorities.append(idx_idx)
    if len(summ_sent_indices) == 0:
        return None, None
    else:
        return summ_sent_indices, priorities

def html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list,
                                    article_sent_tokens, doc_indices=None, lcs_paths_list=None, article_lcs_paths_list=None):
    end_tag = "</mark>"
    out_str = ''

    for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
        try:
            similar_source_indices = similar_source_indices_list[summ_sent_idx]
        except:
            similar_source_indices = []
            a=0
        # lcs_paths = lcs_paths_list[summ_sent_idx]


        for token_idx, token in enumerate(summ_sent):
            insert_string = token + ' '
            for source_indices_idx, source_indices in enumerate(similar_source_indices):
                if source_indices_idx == 0:
                    # print summ_sent_idx
                    try:
                        color = hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                    except:
                        print(summ_sent_idx)
                        print(summary_sent_tokens)
                        print('\n')
                else:
                    color = highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                # if token_idx in lcs_paths[source_indices_idx]:
                # if lcs_paths_list is not None:
                #     lcs_paths_list[summ_sent_idx][source_indices_idx]
                if lcs_paths_list is None or token_idx in lcs_paths_list[summ_sent_idx][source_indices_idx]:
                    insert_string = start_tag_highlight(color) + token + ' ' + end_tag
                    break
                # else:
                #     insert_string = start_tag_highlight(highlight_colors[source_indices_idx]) + token + end_tag
                #     break
            out_str += insert_string
        out_str += '<br><br>'

    cur_token_idx = 0
    cur_doc_idx = 0
    for sent_idx, sent in enumerate(article_sent_tokens):
        if doc_indices is not None:
            if cur_token_idx >= len(doc_indices):
                print("Warning: cur_token_idx is greater than len of doc_indices")
            elif doc_indices[cur_token_idx] != cur_doc_idx:
                cur_doc_idx = doc_indices[cur_token_idx]
                out_str += '<br>'
        summ_sent_indices, priorities = get_idx_for_source_idx(similar_source_indices_list, sent_idx)
        if priorities is None:
            colors = ['black']
            hard_colors = ['black']
        else:
            colors = [highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
            hard_colors = [hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
            # article_lcs_paths = article_lcs_paths_list[summ_sent_idx]
        source_sentence = article_sent_tokens[sent_idx]
        for token_idx, token in enumerate(source_sentence):
            if priorities is None:
                insert_string = token + ' '
            # elif token_idx in article_lcs_paths[priority]:
            else:
                insert_string = token + ' '
                for priority_idx in reversed(list(range(len(priorities)))):
                    summ_sent_idx = summ_sent_indices[priority_idx]
                    priority = priorities[priority_idx]
                    if article_lcs_paths_list is None or token_idx in article_lcs_paths_list[summ_sent_idx][priority]:
                        if priority == 0:
                            insert_string = start_tag_highlight(hard_colors[priority_idx]) + token + ' ' + end_tag
                        else:
                            insert_string = start_tag_highlight(colors[priority_idx]) + token + ' ' + end_tag
            # else:
                # insert_string = start_tag_highlight(highlight_colors[priority]) + token + end_tag
            cur_token_idx += 1
            out_str += insert_string
        out_str += '<br>'
    out_str += '<br>------------------------------------------------------<br><br>'
    return out_str


def get_sent_similarities(summ_sent, article_sent_tokens, vocab, only_rouge_l=False, remove_stop_words=True):
    similarity_matrix = util.rouge_l_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall')
    similarities = np.squeeze(similarity_matrix, 1)

    if not only_rouge_l:
        rouge_l = similarities
        rouge_1 = np.squeeze(util.rouge_1_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', remove_stop_words), 1)
        rouge_2 = np.squeeze(util.rouge_2_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', False), 1)
        similarities = (rouge_l + rouge_1 + rouge_2) / 3.0

    return similarities

def get_simple_source_indices_list(summary_sent_tokens, article_sent_tokens, vocab, sentence_limit, min_matched_tokens, remove_stop_words=True, lemmatize=True, multiple_ssi=False):
    if lemmatize:
        article_sent_tokens_lemma = util.lemmatize_sent_tokens(article_sent_tokens)
        summary_sent_tokens_lemma = util.lemmatize_sent_tokens(summary_sent_tokens)
    else:
        article_sent_tokens_lemma = article_sent_tokens
        summary_sent_tokens_lemma = summary_sent_tokens

    similar_source_indices_list = []
    lcs_paths_list = []
    article_lcs_paths_list = []
    for summ_sent in summary_sent_tokens_lemma:
        remove_lcs = True
        similarities = get_sent_similarities(summ_sent, article_sent_tokens_lemma, vocab)
        if remove_lcs:
            similar_source_indices, lcs_paths, article_lcs_paths = get_similar_source_sents_by_lcs(
                summ_sent, list(range(len(summ_sent))), article_sent_tokens_lemma, vocab, similarities, 0,
                sentence_limit, min_matched_tokens, remove_stop_words=remove_stop_words, multiple_ssi=multiple_ssi)
            similar_source_indices_list.append(similar_source_indices)
            lcs_paths_list.append(lcs_paths)
            article_lcs_paths_list.append(article_lcs_paths)
    deduplicated_similar_source_indices_list = []
    for sim_source_ind in similar_source_indices_list:
        dedup_sim_source_ind = []
        for ssi in sim_source_ind:
            if not (ssi in dedup_sim_source_ind or ssi[::-1] in dedup_sim_source_ind):
                dedup_sim_source_ind.append(ssi)
        deduplicated_similar_source_indices_list.append(dedup_sim_source_ind)
    # for sim_source_ind_idx, sim_source_ind in enumerate(deduplicated_similar_source_indices_list):
    #     if len(sim_source_ind) > 1:
    #         print ' '.join(summary_sent_tokens[sim_source_ind_idx])
    #         print '-----------'
    #         for ssi in sim_source_ind:
    #             for idx in ssi:
    #                 print ' '.join(article_sent_tokens[idx])
    #             print '-------------'
    #         print '\n\n'
    #         a=0
    simple_similar_source_indices = [tuple(sim_source_ind[0]) for sim_source_ind in deduplicated_similar_source_indices_list]
    lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in lcs_paths_list]
    article_lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in article_lcs_paths_list]
    return simple_similar_source_indices, lcs_paths_list, article_lcs_paths_list


def get_similar_source_sents_by_lcs(summ_sent, selection, article_sent_tokens, vocab, similarities, depth, sentence_limit, min_matched_tokens, remove_stop_words=True, multiple_ssi=False):
    remove_unigrams = True
    if sentence_limit == 1:
        if depth > 2:
            return [[]], [[]], [[]]
    elif len(selection) < 3 or depth >= sentence_limit:      # base case: when summary sentence is too short
        return [[]], [[]], [[]]

    all_sent_indices = []
    all_lcs_paths = []
    all_article_lcs_paths = []

    # partial_summ_sent = util.reorder(summ_sent, selection)
    top_sent_indices, top_similarity = get_top_similar_sent(summ_sent, article_sent_tokens, vocab, remove_stop_words, multiple_ssi=multiple_ssi)
    top_similarities = util.reorder(similarities, top_sent_indices)
    top_sent_indices = [x for _, x in sorted(zip(top_similarities, top_sent_indices), key=lambda pair: pair[0])][::-1]
    for top_sent_idx in top_sent_indices:
        # top_sent_idx = top_sent_indices[0]
        if remove_unigrams:
            nonstopword_matches, _ = util.matching_unigrams(summ_sent, article_sent_tokens[top_sent_idx], should_remove_stop_words=remove_stop_words)
            lcs_len, (summ_lcs_path, article_lcs_path) = util.matching_unigrams(summ_sent, article_sent_tokens[top_sent_idx])
        if len(nonstopword_matches) < min_matched_tokens:
            continue
        # new_selection = [selection[idx] for idx in summ_lcs_path]
        # leftover_selection = [val for idx, val in enumerate(selection) if idx not in summ_lcs_path]
        # partial_summ_sent = replace_with_blanks(summ_sent, leftover_selection)
        leftover_selection = [idx for idx in range(len(summ_sent)) if idx not in summ_lcs_path]
        partial_summ_sent = replace_with_blanks(summ_sent, leftover_selection)

        sent_indices, lcs_paths, article_lcs_paths = get_similar_source_sents_by_lcs(
            partial_summ_sent, leftover_selection, article_sent_tokens, vocab, similarities, depth+1,
            sentence_limit, min_matched_tokens, remove_stop_words)   # recursive call

        combined_sent_indices = [[top_sent_idx] + indices for indices in sent_indices]      # append my result to the recursive collection
        combined_lcs_paths = [[summ_lcs_path] + paths for paths in lcs_paths]
        combined_article_lcs_paths = [[article_lcs_path] + paths for paths in article_lcs_paths]

        all_sent_indices.extend(combined_sent_indices)
        all_lcs_paths.extend(combined_lcs_paths)
        all_article_lcs_paths.extend(combined_article_lcs_paths)
    if len(all_sent_indices) == 0:
        return [[]], [[]], [[]]
    return all_sent_indices, all_lcs_paths, all_article_lcs_paths

def get_top_similar_sent(summ_sent, article_sent_tokens, vocab, remove_stop_words=True, multiple_ssi=False):
    try:
        similarities = get_sent_similarities(summ_sent, article_sent_tokens, vocab, remove_stop_words=remove_stop_words)
        top_similarity = np.max(similarities)
    except:
        print(summ_sent)
        print(article_sent_tokens)
        raise
    # sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim == top_similarity]
    if multiple_ssi:
        sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim > top_similarity * 0.75]
    else:
        sent_indices = [np.argmax(similarities)]
    return sent_indices, top_similarity

def replace_with_blanks(summ_sent, selection):
    replaced_summ_sent = [summ_sent[token_idx] if token_idx in selection else '' for token_idx, token in enumerate(summ_sent)]
    return  replaced_summ_sent