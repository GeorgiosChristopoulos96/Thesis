from SPARQLWrapper import SPARQLWrapper, XML, JSON
import wikipediaapi
import csv
from xml.etree import ElementTree as ET
from urllib.parse import unquote

from rdflib import Graph

# Initialize the Wikipedia API for a specific PARENT_metric
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')


def get_philosophers_list():
  sparql = SPARQLWrapper("http://dbpedia.org/sparql")
  sparql_query = """
    PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?philosopher WHERE {
  dbr:List_of_philosophers_born_in_the_centuries_BC ?property ?philosopher.
} LIMIT 100
    """
  sparql.setQuery(sparql_query)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()

  philosophers = [result["philosopher"]["value"] for result in results["results"]["bindings"]]
  return philosophers


def get_rdf_properties(uri):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql_query = f"""
       CONSTRUCT {{
         <{uri}> ?property ?value
       }}
       WHERE {{
         <{uri}> ?property ?value .
         FILTER (lang(?value) = 'en' || datatype(?value) != rdf:langString)
       }}
       """
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(XML)
    results = sparql.query().convert()

    # Serialize the RDF graph to a string in XML format
    rdf_graph = Graph()
    rdf_graph.parse(data=results.serialize(format='application/rdf+xml'), format='xml')
    properties_xml = rdf_graph.serialize(format='pretty-xml')  # This will be bytes

    # Decode bytes to string if necessary
    if isinstance(properties_xml, bytes):
        properties_xml = properties_xml.decode('utf-8')

    return properties_xml



# Step 1: Extract the list of philosophers
philosophers = get_philosophers_list()

# Step 2: For each philosopher, extract and save their RDF properties
dataset = []
for philosopher_uri in philosophers[5:]:
    properties_xml = get_rdf_properties(philosopher_uri)
    dataset.append({
        "uri": philosopher_uri,
        "properties": properties_xml  # This is now a string of XML data
    })

# Save the dataset to a CSV file
with open('philosophers_rdf_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Philosopher URI', 'RDF Properties as XML'])

    for item in dataset:
        # Ensure that properties are a string when writing to CSV
        writer.writerow([item['uri'], item['properties']])

print("Dataset saved to philosophers_rdf_dataset.csv")