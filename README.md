# Context-Aware Conversational AI Interface
This repository contains a conversational AI interface that leverages context-aware language models to provide intelligent and relevant responses to user queries. The interface is built using Python and utilizes various libraries and techniques to enhance the user experience and improve the quality of the AI-generated responses.

![interface](https://github.com/user-attachments/assets/6be32b22-97c7-4237-b498-8444a28f9dae)

## Key Features

- Context-Aware AI: The conversational interface incorporates context-aware AI models that analyze the user's input and relevant historical context to generate more accurate and coherent responses. It considers factors such as the main topic, important details, unique phrases, sentiment, and references to previous messages to provide a tailored response.
- Flexible API Integration: The interface is designed to work with various language models, including self-hosted models like mistral, mixtral, llama3, etc thru Ollama/Llama.cpp API, as well as commercial models like OpenAI's GPT-3. The API calls can be easily adapted to accommodate different models without requiring significant rewrites. (Its currently set up for locally hosted models thru ollama api)
- Vector Database for Context Storage: The interface utilizes a custom vector database (SimpleVectorDatabase) to store and retrieve relevant context information. Each user input and AI response is encoded into a vector representation and stored in the database along with metadata such as message type, context, and timestamp. The database supports efficient similarity search to find relevant messages based on the user's current input.
- Hierarchical Clustering for Context Retrieval: The vector database employs hierarchical clustering to group similar messages together. This allows for efficient retrieval of relevant context based on the similarity of the user's input to the existing messages in the database. The clustering algorithm uses the Ward's linkage method and a user-adjustable clustering threshold to determine the optimal clustering structure.
- Interactive Visualizations: The interface provides interactive visualizations to help users understand the underlying structure and relationships of the stored messages. It includes a dendrogram visualization that represents the hierarchical clustering of the messages and a 3D memory graph visualization with a nebula-like effect and connections between related messages.
- Context Extraction and Vital Context: The interface employs context extraction mechanisms to summarize the key information from each message. It uses the chosen language model to analyze the message and extract relevant details. Additionally, it introduces the concept of "vital context," which identifies the most critical and relevant information from the user's input and historical context that is essential for generating an accurate response.
- Philosophical Query Handling: The interface goes beyond simple question-answering by incorporating philosophical queries into the context sent to the language model. It allows the model to consider more abstract and conceptual aspects of the conversation, enabling it to provide more thoughtful and insightful responses.
- Model Growth and Adaptation: The conversational AI interface is designed to grow and adapt based on user interactions and experiences. As more conversations take place, the vector database expands, and the clustering algorithm refines its grouping of similar messages. This allows the model to continuously improve its understanding of the context and provide more accurate and relevant responses over time.
- Additional Features: The interface includes text-to-speech functionality, Markdown support for rich text formatting, syntax highlighting for code blocks, and a user-friendly graphical interface built with PyQt5.

## Implementation Details
The conversational AI interface is implemented using Python and leverages several libraries and techniques. Here are some key implementation details:

- API Integration: The interface communicates with the chosen language model using the appropriate API. It supports self-hosted models via the Llama.cpp API as well as commercial models like OpenAI's GPT-3. The ApiWorker class is responsible for making API requests in a separate thread to avoid blocking the main interface.
- Vector Database: The SimpleVectorDatabase class represents a custom vector database that stores and retrieves message vectors and their associated metadata. It uses the pickle library for data persistence and supports operations like adding vectors, finding similar messages, adjusting the clustering threshold, and updating the clustering structure.
- Message Encoding: User inputs and AI responses are encoded into vector representations using the MessageEncoder class. It utilizes a pre-trained sentence transformer model (all-MiniLM-L6-v2) to generate the vector representations, which are then stored in the vector database.
- Context Extraction: The extract_context method of the vector database is responsible for extracting the relevant context from each message. It uses the chosen language model to analyze the message and generate a concise summary that captures the key information. The extracted context is cached to avoid redundant API calls.
- Vital Context Extraction: The VitalContextExtractor class is used to extract the vital context for a given user input. It retrieves the most similar messages from the vector database and sends them along with the user input to the language model for analysis. The model generates a summary of the vital context, which is then updated in the vector database and displayed in the context panel.
- Hierarchical Clustering: The vector database employs hierarchical clustering to group similar messages together. It uses the linkage function from the scipy.cluster.hierarchy module to compute the linkage matrix based on the Ward's linkage method and the fcluster function to assign cluster labels to each message based on the specified clustering threshold.
- Interactive Visualizations: The interface provides interactive visualizations using Matplotlib. The plot_dendrogram method generates a dendrogram visualization of the hierarchical clustering, allowing users to explore the similarity between messages at different levels. The update_3d_graph method generates a 3D memory graph visualization with a nebula-like effect and connections between related messages.
- Text-to-Speech: The text-to-speech functionality is implemented using the pyttsx3 library. The TTSWorker class is responsible for converting the AI-generated responses into speech, running in a separate thread to avoid blocking the main interface.
- Markdown Support and Syntax Highlighting: The interface supports the rendering of Markdown-formatted text using the markdown library. It also includes syntax highlighting for code blocks within the chat display using the CodeHighlighter class, which extends the QSyntaxHighlighter class from PyQt5.

## Getting Started
To run the conversational AI interface, follow these steps:

- Clone the repository and install the required dependencies.
- Set up the chosen language model, either by installing and configuring a self-hosted model like Llama.cpp or by obtaining API credentials for a commercial model like OpenAI's GPT-3.
- Update the API URL and model selection in the code to match your chosen language model.
- Run the chat_interface.py script to launch the interface.
- Engage in a conversation with the AI by entering your messages in the chat panel. The AI will generate responses based on the current input and the relevant context retrieved from the vector database.
Explore the interactive visualizations, adjust the clustering sensitivity, and customize the text-to-speech settings as desired.

.cbrwx
