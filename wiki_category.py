# -*- coding: utf-8 -*-
# import wptools
# page = wptools.page('Dog')
# page.get_more()
#
#
# from wikitools import wiki
# from wikitools import category
#
# site = wiki.Wiki("https://en.wikipedia.org/w/api.php?action=query&titles=Stack%20Overflow&prop=categories&clshow=!hidden")
# site.login("username", "password")
# # Create object for "Category:Foo"
# cat = category.Category(site, "Foo")


import wikipedia as w

orig_article = "dog"
reached_categories = []
wiki_top_categories = '''Academic disciplines
Arts‎
Business‎
Concepts‎
Culture‎
Education‎
Entertainment‎
Events‎
Geography‎
Health‎
History‎
Humanities‎
Language‎
Law‎
Life‎
Mathematics
Nature‎
People
Philosophy
Politics‎
Reference‎
Religion‎
Science
Society‎
Sports‎
Technology
Universe‎
World‎'''.replace('\xe2', '').replace('\x80\x8e', '').split('\n')
wiki_top_categories = [cat.strip() for cat in wiki_top_categories]

print wiki_top_categories

def get_top_category(article):
    if article in reached_categories:
        return []
    reached_categories.append(article)
    try:
        page = w.page(article)
    except:
        return []
    categories = page.categories

    top_categories = []
    for cat in categories:
        if cat in wiki_top_categories:
            top_categories.append(cat)
        else:
            top_categories.extend(get_top_category(cat))
    return top_categories

top_categories = get_top_category(orig_article)
print len(top_categories)
print top_categories





























