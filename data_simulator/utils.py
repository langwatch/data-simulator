import pandas as pd
import textwrap

def display_results(results):
    """Display results in a nicely formatted pandas DataFrame."""
    # Extract relevant data
    data = [{
        'ID': item['id'],
        'Query': item['query'],
        'Document': textwrap.shorten(item['document'], width=80, placeholder="..."),
        'Answer': textwrap.shorten(item['answer'], width=120, placeholder="...")
    } for item in results]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set display options for better readability
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 10)
    
    # Display the DataFrame
    print(df.to_string(index=False))
    
    # Reset display options
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.width')
    pd.reset_option('display.max_rows')