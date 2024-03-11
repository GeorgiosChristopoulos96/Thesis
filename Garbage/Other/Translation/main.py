
from SPARQLWrapper import SPARQLWrapper, JSON
import rdflib
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def fetch_data(endpoint,language,LANG_SHORT):
        try:
            sparql = SPARQLWrapper(endpoint)
            sparql.setQuery(f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            SELECT ?resource ?abstract{LANG_SHORT}
            WHERE {{
               ?resource dbo:abstract ?abstract{LANG_SHORT}.
              FILTER(LANG(?abstract{LANG_SHORT}) = "{language}")
            }}

            """)
            sparql.setReturnFormat(JSON) # 60-second timeout
            results[language] = sparql.query().convert()
        except Exception as e:
            print(f"Error fetching data for {language}: {e}")
        return results


    results = {}
    greek_data = fetch_data("http://dbpedia.org/sparql", "ro", 'RO')
    italian_data = ("http:/dbpedia.org/sparql", "it", 'IT')
    # # Match Greek and French abstracts
    # matched_abstracts = {}
    # for resource in greek_abstracts:
    #     if resource in french_abstracts:
    #         matched_abstracts[resource] = {
    #             'el': greek_abstracts[resource],
    #             'fr': french_abstracts[resource]
            #}

    # matched_abstracts now contains resources with abstracts in both Greek and French

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
