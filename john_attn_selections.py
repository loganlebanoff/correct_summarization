
import os
import numpy as np
import json
import glob
from tqdm import tqdm
import nltk
import itertools

gradient_starts = [255, 147]
gradient_ends = [147, 255]
gradient_positions = [1, 0]
original_color = 'ffff93'


def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def create_gradient(match_indices):
    min_ = 147
    max_ = 255
    decimals = np.linspace(min_, max_, len(match_indices))
    hexs = [hex(int(dec))[2:] for dec in decimals]

    '''NEED TO REPLACE THIS PART WITH SOMETHING THAT MAKES THE POSITION DYNAMIC'''
    gradients = ['ff' + h + '93' for h in hexs]
    gradients = [g if match_indices[g_idx] is not None else '' for g_idx, g in enumerate(gradients)]
    return gradients


def create_html(article_lst, match_indices, decoded_lst, abstract_lst, file_idx, summary_sent_tokens, article_sent_tokens):
    colors = create_gradient(match_indices)
    '''<script>document.body.addEventListener("keydown", function (event) {
    if (event.keyCode === 39) {
        window.location.replace("%06d.html");
    }
});</script>'''

    html = '''

<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d.html";
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

''' % (file_idx - 1, file_idx + 1)
    for dec_idx, dec in enumerate(decoded_lst):
        html += '<span style="background-color:#%s;">%s </span>' % (colors[dec_idx], dec)
        if dec == '.' and dec_idx < len(decoded_lst) - 2:
            html += '<br>'

    html += '<br><br>'

    for art_idx, art in enumerate(article_lst):
        if art_idx in match_indices:
            dec_idx = match_indices.index(art_idx)
            color = colors[dec_idx]
        else:
            color = ''
        style = 'style="background-color:#%s;"' % color if color != '' else ''
        html += '<span %s>%s </span>' % (style, art)
        if art == '.' and art_idx < len(article_lst) - 2:
            html += '<br>'

    return html


def process_attn_selections(attn_dir, html_dir):
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    file_names = sorted(glob.glob(os.path.join(attn_dir, '*')))

    for file_idx, file_name in enumerate(tqdm(file_names)):
        with open(file_name) as f:
            data = json.load(f)
        p_gens = flatten_list_of_lists(data['p_gens'])
        article_lst = data['article_lst']
        abstract_lst = data['abstract_str'].strip().split()
        decoded_lst = data['decoded_lst']
        attn_dists = np.array(data['attn_dists'])

        article_lst = [art_word.replace('__', '') for art_word in article_lst]
        decoded_lst = [dec_word.replace('__', '') for dec_word in decoded_lst]
        abstract_lst = [abs_word.replace('__', '') for abs_word in abstract_lst]

        summary_sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in
                               nltk.tokenize.sent_tokenize(' '.join(abstract_lst))]
        decoded_sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in
                               nltk.tokenize.sent_tokenize(' '.join(decoded_lst))]
        article_sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in
                               nltk.tokenize.sent_tokenize(' '.join(article_lst))]

        match_indices = []
        for dec_idx, dec in enumerate(decoded_lst):
            art_match_indices = [art_idx for art_idx, art_word in enumerate(article_lst) if
                                 art_word.replace('__', '') == dec or art_word == dec]
            if len(art_match_indices) == 0:
                match_indices.append(None)
            else:
                art_attns = [attn_dists[dec_idx, art_idx] for art_idx in art_match_indices]
                best_match_idx = art_match_indices[np.argmax(art_attns)]
                match_indices.append(best_match_idx)

        html = create_html(article_lst, match_indices, decoded_lst, [abstract_lst], file_idx, summary_sent_tokens, article_sent_tokens)
        with open(os.path.join(html_dir, '%06d.html' % file_idx), 'wb') as f:
            f.write(html)



def main():

    attn_dir = 'attn_vis_data'
    html_dir = 'extr_vis'

    process_attn_selections(attn_dir, html_dir)


if __name__ == '__main__':
    main()

















