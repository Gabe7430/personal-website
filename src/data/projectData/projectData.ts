export const projects = [
  {
    id: "rnn-lstm-transformer",
    highlight: false,
    title: "RNN, LSTM, Transformer Captioning, and GANs",
    url: "https://github.com/Gabe7430/Projects/tree/main/RNN%2C%20LSTM%2C%20Transformer%20Captioning%2C%20and%20GANs",
    image: { src: "/images/projects/RNN, LSTM, Transformer Captioning, and GANs.png", alt: "RNN, LSTM, Transformer Captioning, and GANs" },
    description: "Exploring cutting-edge deep learning architectures for image captioning and generative modeling. The work focuses on Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Transformer models for generating natural language descriptions of images, alongside Generative Adversarial Networks (GANs) for image synthesis.",
    keyFeatures: [
      "Image Captioning with RNNs and LSTMs",
      "Transformer-based Image Captioning",
      "Generative Adversarial Networks for Image Synthesis",
      "Attention Mechanisms for Improved Captioning",
      "Comparative Analysis of Different Architectures"
    ],
    implementationDetails: "Developed multiple neural network architectures including RNN/LSTM encoders with CNN feature extractors for images, transformer models with self-attention mechanisms, and GANs with generator and discriminator networks. Created comprehensive data preprocessing pipelines, model training procedures, and robust evaluation metrics for comparing different approaches.",
    technologies: ["PyTorch", "TensorFlow", "CNNs", "RNNs", "LSTMs", "Transformers", "GANs", "Attention Mechanisms", "Natural Language Processing", "Computer Vision", "Machine Learning"]
  },
  {
    id: "self-attention-transformers",
    highlight: false,
    title: "Self Attention, Transformers, and Pretraining",
    url: "https://github.com/Gabe7430/Projects/tree/main/Self%20Attention%2C%20Transformers%2C%20and%20Pretraining",
    image: { src: "/images/projects/Self Attention, Transformers, and Pretraining.jpg", alt: "Self Attention, Transformers, and Pretraining" },
    description: "A deep dive into transformer models with self-attention mechanisms for natural language processing. The project examines the architecture and training methodologies of transformer models, with special focus on attention mechanisms, positional encodings, and effective pretraining/fine-tuning approaches.",
    keyFeatures: [
      "Self-Attention Mechanisms",
      "Transformer Encoder-Decoder Architecture",
      "Rotary Position Embeddings (RoPE)",
      "Pretraining and Fine-tuning Pipelines",
      "Multi-head Attention Implementation"
    ],
    implementationDetails: "Built transformer blocks featuring multi-head self-attention, feed-forward networks, layer normalization, and positional encodings. Successfully demonstrated both encoder-only and encoder-decoder architectures, with support for pretraining on large text corpora and fine-tuning for specific downstream tasks.",
    technologies: ["PyTorch", "Transformers", "Self-Attention", "Natural Language Processing", "Pretraining", "Fine-tuning", "Positional Encodings", "Machine Learning"]
  },
  {
    id: "fcn-cnns",
    highlight: false,
    title: "FCN, CNNs, Dropout, Batch Normalization",
    url: "https://github.com/Gabe7430/Projects/tree/main/FCN%2C%20CNNs%2C%20Dropout%2C%20Batch%20Normalization",
    image: { src: "/images/projects/FCN, CNNs, Dropout, Batch Normalization.png", alt: "FCN, CNNs, Dropout, Batch Normalization" },
    description: "Developed convolutional neural networks with regularization techniques for image classification and segmentation. The work investigates optimal CNN architectures and training strategies, with particular emphasis on fully convolutional networks (FCNs), dropout regularization, and batch normalization for improved performance.",
    keyFeatures: [
      "Convolutional Neural Networks for Image Classification",
      "Fully Convolutional Networks for Segmentation",
      "Dropout Regularization",
      "Batch Normalization",
      "Comparative Analysis of Regularization Techniques"
    ],
    implementationDetails: "Designed and constructed various CNN architectures featuring convolutional layers, pooling operations, and different regularization techniques. Conducted experiments on standard image datasets to demonstrate the significant impact of dropout and batch normalization on model performance and training stability.",
    technologies: ["PyTorch", "TensorFlow", "CNNs", "FCNs", "Dropout", "Batch Normalization", "Computer Vision", "Image Classification", "Image Segmentation", "Machine Learning", "Deep Learning"]
  },
  {
    id: "neural-network-yelp",
    highlight: false,
    title: "Neural Network Classifier for Yelp Reviews",
    url: "https://github.com/Gabe7430/Projects/tree/main/Neural%20Network%20Classifer%20for%20Yelp%20Reviews",
    image: { src: "/images/projects/Neural Network Classifer for Yelp Reviews.png", alt: "Neural Network Classifier for Yelp Reviews" },
    description: "A neural network approach to sentiment classification of Yelp reviews. The project leverages deep learning techniques for natural language processing, focusing on accurate sentiment analysis of diverse user-generated content to extract meaningful insights from customer feedback.",
    keyFeatures: [
      "Sentiment Classification of Yelp Reviews",
      "Word Embedding Representations",
      "Neural Network Architecture for Text Classification",
      "Data Preprocessing for NLP Tasks",
      "Model Evaluation and Performance Analysis"
    ],
    implementationDetails: "Created a comprehensive pipeline including text preprocessing, word embedding generation, and a custom neural network classifier. Developed efficient methods for converting unstructured text data into numerical representations, training optimized neural networks on these representations, and rigorously evaluating performance on sentiment classification tasks.",
    technologies: ["PyTorch", "TensorFlow", "Word Embeddings", "Natural Language Processing", "Sentiment Analysis", "Text Classification", "Deep Learning"]
  },
  {
    id: "softmax-svm-knn",
    highlight: false,
    title: "Softmax, SVM, K-Nearest Neighbors",
    url: "https://github.com/Gabe7430/Projects/tree/main/Softmax%2C%20SVM%2C%20K-Nearest%20Neighbors",
    image: { src: "/images/projects/Softmax, SVM, K-Nearest Neighbors.png", alt: "Softmax, SVM, K-Nearest Neighbors" },
    description: "A comparative analysis of three fundamental machine learning algorithms for image classification: Softmax Regression, Support Vector Machines (SVMs), and K-Nearest Neighbors (KNN). The work provides insights into the relative strengths and weaknesses of each approach when applied to standard image datasets under various conditions.",
    keyFeatures: [
      "Softmax Regression Implementation",
      "Support Vector Machine Classifier",
      "K-Nearest Neighbors Algorithm",
      "Comparative Analysis of Classification Techniques",
      "Hyperparameter Tuning and Optimization"
    ],
    implementationDetails: "Developed implementations based on solid mathematical foundations for Softmax regression, SVMs with multiple kernel types, and the KNN algorithm. Created pipelines for feature extraction, model training with cross-validation, and performance evaluation across diverse image classification tasks.",
    technologies: ["Python", "NumPy", "SciPy", "Scikit-learn", "Machine Learning", "Classification Algorithms", "Computer Vision", "Hyperparameter Optimization"]
  },
  {
    id: "ai-traffic-light",
    highlight: true,
    title: "AI Traffic Light Control System",
    url: "https://github.com/Gabe7430/Projects/tree/main/AI%20Traffic%20Light",
    image: { src: "/images/projects/AI Traffic Light.png", alt: "AI Traffic Light Control System" },
    description: "An intelligent traffic light control system powered by reinforcement learning and neural networks for computer vision. The system dynamically optimizes traffic flow by adjusting light timing based on real-time conditions detected through video analysis. By integrating deep reinforcement learning with state-of-the-art object detection, the solution significantly reduces wait times, minimizes congestion, and improves overall traffic efficiency compared to traditional fixed-timing systems.",
    keyFeatures: [
      "Real-time Vehicle and Pedestrian Detection",
      "Reinforcement Learning Agent with DQN",
      "SUMO Traffic Simulation Integration",
      "Adaptive Traffic Light Control",
      "Performance Metrics Tracking",
      "Comparison with Baseline Methods"
    ],
    implementationDetails: "Engineered a traffic light control system with three core components: a computer vision module using YOLOv5 for accurate vehicle detection and tracking, a Deep Q-Network reinforcement learning agent that learns optimal traffic light control policies, and seamless integration with the SUMO traffic simulator for realistic training and testing. Designed state representations including queue lengths and waiting times, with an optimized reward function focused on minimizing total vehicle delay.",
    technologies: ["Python", "PyTorch", "OpenCV", "SUMO Traffic Simulator", "YOLOv5", "Deep Q-Learning", "Object Detection", "Reinforcement Learning", "Computer Vision", "Machine Learning", "Deep Learning"]
  },
  {
    id: "virtual-memory",
    highlight: false,
    title: "Virtual Memory Implementation",
    url: "https://github.com/Gabe7430/Projects/tree/main/Virtual%20Memory",
    image: { src: "/images/projects/Virtual Memory.png", alt: "Virtual Memory Implementation" },
    description: "A custom virtual memory system built in C++ that simulates the core functionality of an operating system's memory management. The solution provides a comprehensive framework for memory mapping, intelligent page fault handling, and optimized physical memory management. The implementation showcases fundamental concepts in operating systems design including address translation, demand paging, and robust memory protection mechanisms.",
    keyFeatures: [
      "Virtual Memory Regions with Configurable Sizes",
      "Page Fault Handling with Custom Signal Handlers",
      "Physical Memory Management",
      "Memory Protection Modes",
      "Dynamic Memory Mapping",
      "Demand Paging",
      "Reference Counting"
    ],
    implementationDetails: "Architected with three primary components: a Virtual Memory Region (VMRegion) module that efficiently manages virtual memory and handles page faults, a Physical Memory Manager (PhysMem) that intelligently allocates and manages physical pages with reference counting for optimal resource utilization, and a Disk Region component providing persistent storage for swapped pages. Leveraged low-level system calls for precise memory mapping and protection controls.",
    technologies: ["C++", "Operating Systems", "Memory Management", "System Programming", "Signal Handling", "POSIX API", "Virtual Memory", "Page Tables"]
  },
  {
    id: "my-own-shell",
    highlight: false,
    title: "My Own Shell (STSH)",
    url: "https://github.com/Gabe7430/Projects/tree/main/My%20Own%20Shell",
    image: { src: "/images/projects/My Own Shell.png", alt: "My Own Shell (STSH)" },
    description: "A custom Unix shell called STSH (Stanford Terminal SHell) built from scratch in C++. The shell delivers core functionality comparable to bash or zsh, with robust support for command execution, input/output redirection, and multi-stage pipelines. The implementation showcases essential operating system concepts including process creation, inter-process communication, and signal handling, all while providing an intuitive and responsive command-line interface.",
    keyFeatures: [
      "Command Execution of Standard Unix Programs",
      "Pipeline Support for Data Flow Between Commands",
      "I/O Redirection to/from Files",
      "Signal Handling (SIGINT, SIGCHLD)",
      "Job Control for Background/Foreground Processes",
      "Custom Command Parser",
      "Comprehensive Error Handling"
    ],
    implementationDetails: "Developed a robust architecture including a command parser for tokenizing and validating user input, process management using fork() and exec() system calls, seamless pipeline implementation with pipes for inter-process communication, comprehensive signal handling for proper process control, and flexible I/O redirection using file descriptors. Implemented careful resource management to ensure proper cleanup of terminated processes and graceful handling of various error conditions.",
    technologies: ["C++", "Operating Systems", "Process Management", "Inter-Process Communication", "Signal Handling", "System Programming", "POSIX API", "Shell Scripting"]
  },
  {
    id: "locks-condition-variables",
    highlight: false,
    title: "Dispatching and Implementing Locks/Condition Variables",
    url: "https://github.com/Gabe7430/Projects/tree/main/Dispatching%20and%20Implementing%20Locks%3ACVs",
    image: { src: "/images/projects/Dispatching and Implementing Locks and CVs.png", alt: "Dispatching and Implementing Locks/Condition Variables" },
    description: "A custom threading library with synchronization primitives built in C++. The library features a thread dispatcher, mutex locks, and condition variables that deliver functionality comparable to standard threading libraries while providing deep insights into concurrency mechanisms. The system elegantly demonstrates core principles of concurrent programming including thread scheduling, reliable mutual exclusion, and precise thread synchronization.",
    keyFeatures: [
      "Thread Dispatcher and Scheduler",
      "Mutex Locks for Critical Sections",
      "Condition Variables for Thread Synchronization",
      "Interrupt Guards for Race Condition Prevention",
      "Thread Management (Creation, Scheduling, Termination)",
      "Context Switching",
      "Deadlock Prevention Mechanisms"
    ],
    implementationDetails: "Engineered comprehensive thread management with efficient creation, scheduling, and low-overhead context switching between threads. Designed mutex locks featuring waiting queues for blocked threads and owner tracking to prevent recursive locking issues. Implemented condition variables with atomic wait and notification mechanisms that seamlessly release mutexes and block threads. Added robust interrupt guards to eliminate race conditions during critical operations.",
    technologies: ["C++", "Operating Systems", "Concurrency", "Thread Programming", "Synchronization Primitives", "Context Switching", "Mutual Exclusion", "Condition Variables"]
  },
  {
    id: "reinforcement-learning-stock",
    highlight: true,
    title: "Reinforcement Learning Stacking Strategy for Stock Trading",
    url: "https://github.com/Gabe7430/Projects/tree/main/Reinforcement%20Learning%20Stacking%20Strategy%20for%20Stock%20Trading",
    image: { src: "/images/projects/Reinforcement Learning Stacking Strategy for Stock Trading.png", alt: "Reinforcement Learning Stacking Strategy for Stock Trading" },
    description: "A stock trading strategy that seamlessly combines supervised learning with novel reinforcement learning techniques. The system leverages ensemble deep reinforcement learning (DRL) to make intelligent trading decisions based on comprehensive market data, fundamental analysis, and technical indicators. The strategy is designed to optimize portfolio performance while effectively managing risk across highly dynamic market conditions.",
    keyFeatures: [
      "Ensemble Deep Reinforcement Learning for Trading",
      "Fundamental Analysis Integration",
      "Technical Indicator Processing",
      "Portfolio Optimization",
      "Risk Management",
      "Market Simulation Environment",
      "Performance Benchmarking"
    ],
    implementationDetails: "Developed a realistic market environment simulator, feature engineering for financial data, and multiple DRL agents with different architectures (DQN, PPO, A2C). Implemented an ensemble mechanism for combining agent decisions and a powerful portfolio optimization module. Trained the system on extensive historical market data and rigorously evaluated it against established benchmark trading strategies.",
    technologies: ["Python", "TensorFlow", "PyTorch", "Reinforcement Learning", "Deep Learning", "Financial Analysis", "Time Series Processing", "Portfolio Optimization", "Ensemble Methods", "Machine Learning"]
  },
  {
    id: "mountaincar-rl",
    highlight: false,
    title: "Controlling MountainCar with Value Iteration, MDPs, Q-Learning, and Safe Exploration",
    url: "https://github.com/Gabe7430/Projects/tree/main/Controlling%20MountainCar%20with%20Value%20Iteration%2C%20MDPs%2C%20Q-Learning%2C%20and%20Safe%20Exploration",
    image: { src: "/images/projects/Controlling MountainCar with Value Iteration, MDPs, Q-Learning, and Safe Exploration.png", alt: "Controlling MountainCar with Value Iteration, MDPs, Q-Learning, and Safe Exploration" },
    description: "A suite of reinforcement learning algorithms for solving the MountainCar environment, a challenging control problem where an underpowered car must navigate up a steep hill. The work explores various RL approaches including value iteration, Markov Decision Processes (MDPs), and Q-learning, with particular emphasis on implementing safe and efficient exploration strategies.",
    keyFeatures: [
      "Value Iteration for MDP Solving",
      "Q-Learning Implementation",
      "Safe Exploration Strategies",
      "MDP Modeling and Discretization",
      "Performance Comparison of Different Algorithms",
      "Visualization of Learning Progress"
    ],
    implementationDetails: "Leveraged the Bellman Equation in value iteration to solve the discretized MDP, implemented Q-learning with multiple exploration strategies (ε-greedy, optimistic initialization), and designed safety mechanisms to prevent catastrophic failures during exploration. Developed comprehensive environment modeling, state discretization techniques, reward shaping methods, and interactive visualization tools for policy evaluation.",
    technologies: ["Python", "OpenAI Gym", "NumPy", "Reinforcement Learning", "Markov Decision Processes", "Q-Learning", "Dynamic Programming", "Control Theory"]
  },
  {
    id: "model-free-q-learning",
    highlight: false,
    title: "Model Free Q-Learning Agent for Optimal Policies in Different MDPs",
    url: "https://github.com/Gabe7430/Projects/tree/main/Model%20Free%20Q-Learning%20Agent%20to%20Generate%20Optimal%20Policies%20for%203%20Different%20MDPs",
    image: { src: "/images/projects/Model Free Q-Learning Agent to Generate Optimal Policies for 3 Different MDPs.png", alt: "Model Free Q-Learning Agent for Optimal Policies in Different MDPs" },
    description: "A versatile model-free Q-learning agent that generates optimal policies across three distinct Markov Decision Processes (MDPs). The solution elegantly demonstrates how a single reinforcement learning algorithm can adapt to diverse environments with varying dynamics and reward structures, showcasing the flexibility and power of Q-learning.",
    keyFeatures: [
      "Model-Free Q-Learning Implementation",
      "Adaptation to Multiple MDP Environments",
      "Exploration Strategy Optimization",
      "Learning Rate and Discount Factor Tuning",
      "Policy Visualization and Evaluation",
      "Convergence Analysis"
    ],
    implementationDetails: "Engineered an adaptable Q-learning agent capable of seamless application across different MDP environments. Integrated mechanisms for balancing exploration-exploitation tradeoffs, dynamic learning rate scheduling, experience replay for improved sample efficiency, and interactive visualization tools for comprehensive policy and value function analysis.",
    technologies: ["Python", "NumPy", "Matplotlib", "Reinforcement Learning", "Q-Learning", "Markov Decision Processes", "Policy Optimization"]
  },
  {
    id: "wordvec-dependency-parsing",
    highlight: false,
    title: "WordVec and Dependency Parsing",
    url: "https://github.com/Gabe7430/Projects/tree/main/WordVec%20and%20Dependency%20Parsing",
    image: { src: "/images/projects/WordVec and Dependency Parsing.png", alt: "WordVec and Dependency Parsing" },
    description: "A neural dependency parser leveraging word embeddings and a feed-forward neural network architecture. The solution focuses on transition-based dependency parsing with an arc-standard system, where a feed-forward neural network accurately predicts the next transition (shift, left-arc, or right-arc) based on features extracted from the current parser state.",
    keyFeatures: [
      "Neural Dependency Parser Implementation",
      "Word Embedding Utilization",
      "Arc-Standard Transition System",
      "Feed-Forward Neural Network Architecture",
      "Minibatch Processing for Efficiency",
      "PyTorch Implementation"
    ],
    implementationDetails: "Developed a robust transition-based parser with shift, left-arc, and right-arc operations, a partial parse data structure to track stack, buffer, and dependencies, and a feed-forward neural network for accurate transition prediction. Incorporated word embedding lookup for feature extraction and implemented minibatch processing for significantly improved parsing efficiency.",
    technologies: ["Python", "PyTorch", "Natural Language Processing", "Word Embeddings", "Dependency Parsing", "Neural Networks", "Transition-Based Parsing"]
  },
  {
    id: "exploring-word-vectors",
    highlight: false,
    title: "Exploring Word Vectors",
    url: "https://github.com/Gabe7430/Projects/tree/main/Exploring%20Word%20Vectors",
    image: { src: "/images/projects/Exploring Word Vectors.png", alt: "Exploring Word Vectors" },
    description: "An in-depth exploration of word embedding techniques for semantic representation of text. The work investigates how distributed representations like Word2Vec and GloVe effectively capture complex semantic relationships between words, demonstrating their practical utility in tasks such as semantic similarity calculation and solving word analogies.",
    keyFeatures: [
      "Word Embedding Analysis and Visualization",
      "Semantic Similarity Calculations",
      "Word Analogy Solving",
      "Comparison of Different Embedding Techniques",
      "Dimensionality Reduction for Visualization",
      "Application to NLP Tasks"
    ],
    implementationDetails: "Created comprehensive tools for loading and analyzing pre-trained word embeddings, computing precise semantic similarity between words, solving complex word analogies (e.g., 'king' - 'man' + 'woman' = 'queen'), and visualizing intricate word relationships in lower-dimensional spaces using techniques like t-SNE and PCA.",
    technologies: ["Python", "NumPy", "Word2Vec", "GloVe", "Natural Language Processing", "Semantic Analysis", "Dimensionality Reduction", "Vector Space Models"]
  },
  {
    id: "information-retrieval-gutenberg",
    highlight: false,
    title: "Information Retrieval System for Project Gutenberg",
    url: "https://github.com/Gabe7430/Projects/tree/main/Information%20Retrieval%20System%20for%20Project%20Gutenberg%20(eBooks)",
    image: { src: "/images/projects/Information Retrieval System for Project Gutenberg (eBooks).png.webp", alt: "Information Retrieval System for Project Gutenberg" },
    description: "A search engine for retrieving relevant documents from Project Gutenberg's vast collection of eBooks. The system employs information retrieval techniques to index, search, and rank documents based on their relevance to user queries, offering an intuitive way to discover information within a large corpus of literary texts.",
    keyFeatures: [
      "Inverted Index Construction",
      "TF-IDF Scoring Implementation",
      "Query Processing and Expansion",
      "Relevance Ranking Algorithms",
      "Document Preprocessing Pipeline",
      "Search Result Presentation"
    ],
    implementationDetails: "Built an optimized inverted index for rapid document retrieval, implemented TF-IDF scoring to accurately measure document relevance, developed query processing with stemming and stopword removal, and created ranking algorithms to sort results by relevance. Designed a comprehensive document preprocessing pipeline and an intuitive user interface for submitting queries and viewing organized results.",
    technologies: ["Python", "NLTK", "Information Retrieval", "Inverted Index", "TF-IDF", "Text Processing", "Search Algorithms", "Ranking Algorithms"]
  },
  {
    id: "synthetic-dataset-pose-estimation",
    highlight: true,
    title: "Synthetic Dataset Generation and Enhanced 6D Pose Estimation",
    url: "https://github.com/Gabe7430/Projects/tree/main/Synthetic%20Dataset%20Generation%20and%20Enhanced%206D%20Pose%20Estimation",
    image: { src: "/images/projects/Synthetic Dataset Generation and Enhanced 6D Pose Estimation.png", alt: "Synthetic Dataset Generation and Enhanced 6D Pose Estimation" },
    description: "A system for generating high-quality synthetic datasets for 6D pose estimation of objects in robotics and computer vision applications. The solution focuses on creating photorealistic simulated environments using ROS2 and Gazebo, capturing synthetic RGB and depth images along with precise ground truth pose information essential for training robust deep learning models.",
    keyFeatures: [
      "Synthetic Data Generation for 6D Pose Estimation",
      "ROS2 and Gazebo Integration",
      "Auto and Manual World Generation",
      "Multi-Modal Data Capture (RGB, Depth, Segmentation)",
      "Domain Randomization Techniques",
      "Deep Learning for Pose Estimation"
    ],
    implementationDetails: "Engineered a comprehensive ROS2-based framework for simulation, leveraged the Gazebo physics engine for realistic object interactions, developed automated capture systems for perfectly synchronized RGB and depth images, created tools for generating pixel-perfect segmentation masks, and built precise 6D pose ground truth recording mechanisms. Incorporated domain randomization techniques to significantly increase data diversity and improve model generalization.",
    technologies: ["Python", "ROS2", "Gazebo", "Computer Vision", "Deep Learning", "6D Pose Estimation", "Synthetic Data Generation", "Domain Randomization"]
  },
  {
    id: "multi-agent-pacman",
    highlight: false,
    title: "Multi-Agent Pac-Man using Minimax and Expectimax",
    url: "https://github.com/Gabe7430/Projects/tree/main/Multia-Agent%20Pac-Man%20using%20Minimax%20and%20Expectimax",
    image: { src: "/images/projects/Multia-Agent Pac-Man using Minimax and Expectimax.png", alt: "Multi-Agent Pac-Man using Minimax and Expectimax" },
    description: "An intelligent agent system using adversarial search algorithms for the multi-agent Pac-Man game. The work explores how algorithms like Minimax, Alpha-Beta Pruning, and Expectimax can create strategic agents capable of making optimal decisions in complex competitive environments with multiple actors and uncertain outcomes.",
    keyFeatures: [
      "Minimax Algorithm Implementation",
      "Alpha-Beta Pruning for Efficiency",
      "Expectimax for Stochastic Environments",
      "Evaluation Function Design",
      "Multi-Agent Game Environment",
      "Performance Analysis of Different Algorithms"
    ],
    implementationDetails: "Developed the Minimax algorithm for deterministic adversarial search scenarios, implemented Alpha-Beta pruning for dramatically improved computational efficiency, and created Expectimax algorithms for effectively handling probabilistic ghost behavior. Designed evaluation functions that intelligently consider food locations, ghost positions, and various other game state features to optimally guide the Pac-Man agent's strategic decisions.",
    technologies: ["Python", "Artificial Intelligence", "Adversarial Search", "Minimax", "Alpha-Beta Pruning", "Expectimax", "Game Theory", "Heuristic Evaluation"]
  },
  {
    id: "route-planning-astar",
    highlight: false,
    title: "Route Planning and Finding Shortest Paths with A-Star",
    url: "https://github.com/Gabe7430/Projects/tree/main/Route%20Planning%20and%20Finding%20Shortest%20Paths%20with%20A-Star",
    image: { src: "/images/projects/Route Planning and Finding Shortest Paths with A-Star.png", alt: "Route Planning and Finding Shortest Paths with A-Star" },
    description: "An intelligent route planning system leveraging search algorithms to find optimal paths through complex maps. The solution centers on the A* search algorithm to efficiently discover shortest paths between locations, with comprehensive support for multiple waypoints and custom heuristics that significantly improve search efficiency in large-scale navigation problems.",
    keyFeatures: [
      "A* Search Algorithm Implementation",
      "Shortest Path Finding Between Locations",
      "Waypoints Support for Route Planning",
      "Custom Heuristics for Search Efficiency",
      "Interactive Map Visualization",
      "Real-world Map Data Integration"
    ],
    implementationDetails: "Developed an optimized A* search algorithm with custom heuristics like straight-line distance, created a flexible ShortestPathProblem class for finding paths to destinations, and engineered a WaypointsShortestPathProblem extension for complex routing through multiple specified waypoints. Integrated the system with OpenStreetMap data and built interactive visualization tools for displaying routes clearly and intuitively.",
    technologies: ["Python", "A* Search Algorithm", "Graph Theory", "Heuristic Search", "OpenStreetMap", "Plotly", "Route Planning", "Pathfinding Algorithms"]
  },
  {
    id: "svd-pca-kmeans-recommendation",
    highlight: false,
    title: "SVD, PCA, K-Means (On Spark), and Latent Features for Recommendation",
    url: "https://github.com/Gabe7430/Projects/tree/main/SVD%2C%20PCA%2C%20K-Means%20(On%20Spark)%2C%20and%20Latent%20Features%20for%20Recommendation",
    image: { src: "/images/projects/SVD, PCA, K-Means (On Spark), and Latent Features for Recommendation.png.webp", alt: "SVD, PCA, K-Means (On Spark), and Latent Features for Recommendation" },
    description: "A recommendation system utilizing matrix factorization and clustering techniques. The work explores how Singular Value Decomposition (SVD), Principal Component Analysis (PCA), and K-Means clustering effectively identify hidden patterns in user-item interaction data to generate highly personalized and relevant recommendations.",
    keyFeatures: [
      "Singular Value Decomposition Implementation",
      "Principal Component Analysis for Dimensionality Reduction",
      "K-Means Clustering on Apache Spark",
      "Collaborative Filtering Techniques",
      "Matrix Factorization for Recommendation",
      "Latent Feature Extraction and Analysis"
    ],
    implementationDetails: "Engineered a comprehensive solution featuring SVD for efficient matrix factorization, PCA for effective dimensionality reduction, and K-Means clustering implemented on Apache Spark for exceptional scalability. Developed collaborative filtering algorithms including both user-user and item-item approaches, with optimized methods for matrix initialization, decomposition, and generating accurate recommendations based on extracted latent features.",
    technologies: ["Python", "NumPy", "Apache Spark", "Recommendation Systems", "Matrix Factorization", "SVD", "PCA", "K-Means Clustering", "Collaborative Filtering"]
  },
  {
    id: "decision-trees-gda-adaboost",
    highlight: false,
    title: "Decision Trees, GDA, AdaBoost, Spam Classification",
    url: "https://github.com/Gabe7430/Projects/tree/main/Decision%20Trees%2C%20GDA%2C%20AdaBoost%2C%20Spam%20Classification",
    image: { src: "/images/projects/Decision Trees, GDA, AdaBoost, Spam Classification.png", alt: "Decision Trees, GDA, AdaBoost, Spam Classification" },
    description: "A robust spam detection system utilizing multiple classification algorithms. The solution explores and compares the effectiveness of decision trees, Gaussian Discriminant Analysis (GDA), and AdaBoost ensemble methods for accurately classifying emails as spam or legitimate based on content analysis and feature extraction.",
    keyFeatures: [
      "Decision Tree Implementation and Pruning",
      "Gaussian Discriminant Analysis (GDA) Classifier",
      "AdaBoost Ensemble Method",
      "Spam Email Classification",
      "Feature Extraction from Text Data",
      "Model Evaluation and Comparison"
    ],
    implementationDetails: "Created decision trees with optimized information gain splitting criteria and effective pruning techniques, implemented GDA with maximum likelihood estimation for accurate class-conditional distributions, and developed AdaBoost algorithms for intelligently combining weak learners into a powerful ensemble classifier. Built comprehensive pipelines for feature extraction from email text, implemented rigorous cross-validation for model selection, and designed thorough performance evaluation metrics.",
    technologies: ["Python", "NumPy", "SciPy", "Machine Learning", "Classification Algorithms", "Decision Trees", "Gaussian Models", "Ensemble Methods", "Text Classification"]
  },
  {
    id: "ml-classification-covid",
    highlight: false,
    title: "ML Classification for Disaster Aid and Reddit Comments for COVID",
    url: "https://github.com/Gabe7430/Projects/tree/main/ML%20Classification%20for%20Disaster%20Aid%20and%20Reddit%20Comments%20for%20COVID",
    image: { src: "/images/projects/ML Classification for Disaster Aid and Reddit Comments for COVID.jpg", alt: "ML Classification for Disaster Aid and Reddit Comments for COVID" },
    description: "An application of machine learning classification techniques to two critical datasets: medical triage data and Reddit comments related to COVID-19. The work demonstrates how algorithms like Logistic Regression and Naive Bayes can provide valuable medical decision support and insightful social media content analysis during public health crises.",
    keyFeatures: [
      "Logistic Regression Implementation",
      "Naive Bayes Classifier",
      "Medical Triage Classification",
      "COVID-19 Related Social Media Analysis",
      "Text Preprocessing Pipeline",
      "Feature Engineering for Medical and Text Data"
    ],
    implementationDetails: "Developed optimized logistic regression with gradient descent optimization and Naive Bayes with adaptive smoothing for robust text classification. Created specialized data preprocessing pipelines for both structured medical data and unstructured text data, implemented feature extraction using bag-of-words and TF-IDF representations, and designed custom evaluation metrics specifically calibrated for imbalanced datasets.",
    technologies: ["Python", "NumPy", "Pandas", "Scikit-learn", "NLTK", "Machine Learning", "Text Classification", "Medical Data Analysis", "Social Media Mining"]
  },
  {
    id: "movie-ratings-sentiment",
    highlight: false,
    title: "Predicting Movie Ratings, Sentiment Classification, and Toxicity Classification",
    url: "https://github.com/Gabe7430/Projects/tree/main/Predicting%20Movie%20Ratings%2C%20Sentiment%20Classification%2C%20and%20Toxicity%20Classification",
    image: { src: "/images/projects/Predicting Movie Ratings, Sentiment Classification, and Toxicity Classification.png", alt: "Predicting Movie Ratings, Sentiment Classification, and Toxicity Classification" },
    description: "A suite of text classification models for sentiment analysis, toxicity detection, and movie rating prediction. The work explores advanced feature extraction techniques, optimized stochastic gradient descent algorithms, and n-gram features to accurately analyze and classify diverse text data from movie reviews and online comments.",
    keyFeatures: [
      "Movie Rating Prediction from Reviews",
      "Sentiment Analysis of Text Data",
      "Toxicity Classification for Online Comments",
      "Feature Extraction with N-grams",
      "Stochastic Gradient Descent Optimization",
      "Model Evaluation and Error Analysis"
    ],
    implementationDetails: "Built comprehensive text preprocessing pipelines with tokenization and normalization techniques, developed feature extraction systems using n-gram models and TF-IDF weighting, trained classification models with optimized stochastic gradient descent, and created regression models for accurate rating prediction. Incorporated systematic hyperparameter tuning, robust cross-validation protocols, and detailed performance analysis across multiple text domains.",
    technologies: ["Python", "NumPy", "Pandas", "Scikit-learn", "NLTK", "Natural Language Processing", "Text Classification", "Sentiment Analysis", "Stochastic Gradient Descent"]
  },
  {
    id: "from-language-to-logic",
    highlight: false,
    title: "From Language to Logic",
    url: "https://github.com/Gabe7430/Projects/tree/main/From%20Language%20to%20Logic",
    image: { src: "/images/projects/From Language to Logic.png", alt: "From Language to Logic" },
    description: "A sophisticated system for translating natural language statements into formal logical representations. The work explores the complex process of semantic parsing, where everyday language is precisely converted into structured logical expressions that enable powerful automated reasoning and inference capabilities.",
    keyFeatures: [
      "Natural Language to Logic Translation",
      "Semantic Parsing Implementation",
      "Logical Inference Engine",
      "Knowledge Representation",
      "Grammar-based Parsing",
      "Compositional Semantics"
    ],
    implementationDetails: "Engineered a comprehensive pipeline for converting natural language to logical forms, featuring robust syntactic parsing, semantic interpretation rules, and precise logical form construction. Developed an extensible knowledge base for efficiently storing facts, built a powerful inference engine for reasoning with logical forms, and created rigorous evaluation methods using human-annotated logical representations as ground truth.",
    technologies: ["Python", "Natural Language Processing", "Formal Logic", "Semantic Parsing", "Knowledge Representation", "Automated Reasoning", "Computational Linguistics"]
  },
  {
    id: "glass-bridge-blender",
    highlight: false,
    title: "Glass Bridge Scene from Squid Games (Blender)",
    url: "https://github.com/Gabe7430/Projects/tree/main/Glass%20Bridge%20Scene%20from%20Squid%20Games%20(Blender)",
    image: { src: "/images/projects/Glass Bridge Scene from Squid Games (Blender).png", alt: "Glass Bridge Scene from Squid Games (Blender)" },
    description: "A detailed recreation of the iconic glass bridge scene from the popular TV show Squid Game using Blender. The work showcases 3D modeling, realistic texturing, dramatic lighting, and professional rendering techniques to create a photorealistic version of this tense and visually striking environment.",
    keyFeatures: [
      "3D Modeling in Blender",
      "Photorealistic Glass Material Creation",
      "Advanced Lighting Setup",
      "Environment Design",
      "Camera Work and Composition",
      "Post-processing Effects"
    ],
    implementationDetails: "Created meticulously detailed 3D models of the bridge structure, developed physically-based materials for realistic glass and other surfaces, designed a professional three-point lighting setup with global illumination, established strategic camera positioning for dramatic angles, and applied post-processing effects including depth of field and color grading for cinematic quality.",
    technologies: ["Blender", "3D Modeling", "Texturing", "Lighting", "Rendering", "Material Design", "Post-processing", "Computer Graphics"]
  },
  {
    id: "car-tracking",
    highlight: true,
    title: "Car Tracking",
    url: "https://github.com/Gabe7430/Projects/tree/main/Car%20Tracking",
    image: { src: "/images/projects/Car Tracking.png", alt: "Car Tracking" },
    description: "A robust computer vision system for vehicle tracking in video footage. The solution employs object detection, feature extraction, and tracking techniques to accurately identify and follow multiple vehicles across video frames, with practical applications in traffic monitoring, autonomous driving, and intelligent surveillance systems.",
    keyFeatures: [
      "Vehicle Detection in Video Footage",
      "Feature-based Tracking Algorithms",
      "Motion Estimation and Prediction",
      "Occlusion Handling",
      "Multi-vehicle Tracking",
      "Performance Evaluation Metrics"
    ],
    implementationDetails: "Developed a comprehensive system featuring object detection using optimized pre-trained models and adaptive background subtraction, robust feature extraction for accurate vehicle representation, advanced tracking algorithms including Kalman filtering and optical flow, data association methods for maintaining consistent vehicle identities across frames, and thorough performance evaluation using industry-standard metrics like MOTA and MOTP.",
    technologies: ["Python", "OpenCV", "Computer Vision", "Object Detection", "Object Tracking", "Feature Extraction", "Motion Analysis", "Video Processing"]
  },
  {
    id: "huffman-coding",
    highlight: false,
    title: "Huffman Coding",
    url: "https://github.com/Gabe7430/Projects/tree/main/Huffman%20Coding",
    image: { src: "/images/projects/Huffman Coding.png", alt: "Huffman Coding" },
    description: "Huffman coding implementation for lossless data compression. The work demonstrates how optimized variable-length prefix codes can be generated based on symbol frequency analysis, resulting in significant space savings when compressing text and various other file types.",
    keyFeatures: [
      "Huffman Tree Construction",
      "Variable-length Code Generation",
      "File Compression and Decompression",
      "Frequency Analysis",
      "Binary I/O Operations",
      "Compression Ratio Analysis"
    ],
    implementationDetails: "Built a complete compression pipeline featuring frequency analysis of symbols in the input data, optimized priority queue-based Huffman tree construction, dynamic code table generation for mapping symbols to bit sequences, high-performance bit-level I/O operations for writing and reading compressed data, and comprehensive analysis of compression ratios across diverse file types and sizes.",
    technologies: ["C++", "Python", "Data Structures", "Algorithms", "Greedy Algorithms", "Binary Trees", "Priority Queues", "Data Compression"]
  },
  {
    id: "course-scheduling-csp",
    highlight: false,
    title: "Course Scheduling and Residency Hours Scheduling as CSPs",
    url: "https://github.com/Gabe7430/Projects/tree/main/Course%20Scheduling%20and%20Residency%20Hours%20Scheduling%20as%20CSPs",
    image: { src: "/images/projects/Course Scheduling and Residency Hours Scheduling as CSPs.jpg", alt: "Course Scheduling and Residency Hours Scheduling as CSPs" },
    description: "A practical application of constraint satisfaction problems (CSPs) to solve complex scheduling challenges. The solution demonstrates how course scheduling and medical residency hours allocation can be elegantly modeled with variables, domains, and constraints, then solved using advanced techniques like constraint propagation and intelligent backtracking search.",
    keyFeatures: [
      "CSP Formulation of Scheduling Problems",
      "Constraint Propagation Algorithms",
      "Backtracking Search Implementation",
      "Local Search for Optimization",
      "Course Scheduling Application",
      "Medical Residency Hours Allocation"
    ],
    implementationDetails: "Designed flexible CSP representations with intuitive variables for courses/shifts, comprehensive domains of possible times/assignments, and realistic constraints on resources and requirements. Implemented algorithms for arc consistency, forward checking, dynamic variable ordering heuristics, and backtracking search with constraint propagation. Applied local search methods for further optimization after identifying feasible solutions.",
    technologies: ["Python", "Constraint Satisfaction Problems", "Constraint Programming", "Search Algorithms", "Optimization", "Scheduling Algorithms", "Heuristic Search"]
  },
  {
    id: "kmeans-em-pca-nn",
    highlight: false,
    title: "K-means, Semi-Supervised EM, PCA, and Simple Neural Network",
    url: "https://github.com/Gabe7430/Projects/tree/main/K_means%2C%20Semi-Supervised%20EM%2C%20PCA%2C%20and%20Simple%20Neural%20Network",
    image: { src: "/images/projects/K_means, Semi-Supervised EM, PCA, and Simple Neural Network.png", alt: "K-means, Semi-Supervised EM, PCA, and Simple Neural Network" },
    description: "A comprehensive toolkit of clustering, dimensionality reduction, and neural network techniques for unsupervised and semi-supervised learning tasks. The work explores K-means clustering, the Expectation-Maximization (EM) algorithm, Principal Component Analysis (PCA), and neural network architectures for effective pattern recognition and insightful data analysis.",
    keyFeatures: [
      "K-means Clustering Implementation",
      "Semi-Supervised Expectation-Maximization",
      "Principal Component Analysis",
      "Simple Neural Network Architecture",
      "Dimensionality Reduction",
      "Cluster Evaluation and Visualization"
    ],
    implementationDetails: "Developed K-means clustering with multiple initialization strategies for improved convergence, implemented the EM algorithm with Gaussian mixture models and novel semi-supervised extensions, created an efficient PCA implementation for dimensionality reduction using eigenvalue decomposition, and built a neural network with optimized backpropagation. Incorporated comprehensive data preprocessing pipelines, interactive visualization tools for clusters and principal components, and robust evaluation metrics for assessing clustering quality.",
    technologies: ["Python", "NumPy", "SciPy", "Matplotlib", "Machine Learning", "Clustering", "Dimensionality Reduction", "Neural Networks", "Unsupervised Learning"]
  },
  {
    id: "gradient-descent-regression",
    highlight: false,
    title: "Gradient Descent, Linear Regression, Poisson Regression",
    url: "https://github.com/Gabe7430/Projects/tree/main/Gradient%20Descent%2C%20Linear%20Regression%2C%20Poisson%20Regression",
    image: { src: "/images/projects/Gradient Descent, Linear Regression, Poisson Regression.png", alt: "Gradient Descent, Linear Regression, Poisson Regression" },
    description: "A suite of optimization and regression techniques for high-quality predictive modeling. The work explores various gradient descent optimization approaches, linear regression methods for continuous outcomes, and specialized Poisson regression for count data, demonstrating effective approaches to fitting statistical models to diverse datasets.",
    keyFeatures: [
      "Gradient Descent Optimization",
      "Linear Regression Implementation",
      "Poisson Regression for Count Data",
      "Regularization Techniques",
      "Model Evaluation and Diagnostics",
      "Convergence Analysis"
    ],
    implementationDetails: "Created optimized implementations of gradient descent with multiple variants (batch, stochastic, mini-batch), developed linear regression models using both least squares and maximum likelihood estimation approaches, built Poisson regression with log link function for count data, and incorporated regularization techniques including ridge and lasso. Designed adaptive learning rate scheduling, robust convergence criteria, and comprehensive diagnostic tools for thorough model evaluation.",
    technologies: ["Python", "NumPy", "Pandas", "Matplotlib", "Statistical Modeling", "Regression Analysis", "Optimization Algorithms", "Predictive Modeling"]
  },
  {
    id: "into-the-void",
    highlight: false,
    title: "Into the Void (C Programming with Pointers)",
    url: "https://github.com/Gabe7430/Projects/tree/main/Into%20the%20Void",
    image: { src: "/images/projects/Into the Void*.png", alt: "Into the Void (C Programming with Pointers)" },
    description: "A deep dive into memory management and pointer manipulation in C. The work demonstrates essential low-level programming concepts including dynamic memory allocation, pointer arithmetic, and memory safety techniques, providing comprehensive insights into memory management fundamentals critical for effective systems programming.",
    keyFeatures: [
      "Dynamic Memory Allocation",
      "Pointer Arithmetic and Manipulation",
      "Memory Safety Techniques",
      "Data Structure Implementation",
      "Memory Leak Prevention",
      "Low-level Optimization"
    ],
    implementationDetails: "Developed custom memory management functions optimized for different use cases, created efficient data structures using pointers (linked lists, trees, graphs), implemented robust memory safety checks to prevent common errors like buffer overflows and use-after-free, and designed techniques for detecting and preventing memory leaks. Included practical examples of memory usage patterns and documented common pitfalls to avoid in production code.",
    technologies: ["C", "Systems Programming", "Memory Management", "Pointers", "Data Structures", "Debugging", "Low-level Programming"]
  },
  {
    id: "movie-recommender-chatbot",
    highlight: false,
    title: "Movie Recommender Chatbot",
    url: "https://github.com/Gabe7430/Projects/tree/main/Movie%20Recommender%20Chatbot",
    image: { src: "/images/projects/Movie Recommender Chatbot.png", alt: "Movie Recommender Chatbot" },
    description: "An intelligent conversational agent specialized in movie recommendations. The system seamlessly combines natural language processing for understanding nuanced user preferences with recommendation algorithms to suggest highly relevant movies, all within an engaging dialogue-based interface that closely simulates natural human conversation.",
    keyFeatures: [
      "Conversational Interface for Movie Recommendations",
      "Natural Language Understanding",
      "Recommendation Algorithm Implementation",
      "User Preference Modeling",
      "Dialogue Management",
      "Movie Database Integration"
    ],
    implementationDetails: "Built advanced natural language processing components for accurate intent recognition and entity extraction, created a context-aware dialogue management system for maintaining coherent conversations, developed user preference modeling based on both explicit and implicit feedback signals, and implemented hybrid recommendation algorithms combining collaborative filtering with content-based approaches. Integrated with comprehensive movie databases for rich metadata and designed an intuitive user interface for seamless text-based interaction.",
    technologies: ["Python", "Natural Language Processing", "Recommendation Systems", "Dialogue Systems", "Machine Learning", "Collaborative Filtering", "User Modeling", "Conversational AI"]
  },
  {
    id: "legal-case-analysis",
    highlight: false,
    title: "Legal Case Analysis",
    url: "https://github.com/Gabe7430/Projects/tree/main/Legal%20Case%20Analysis",
    image: { src: "/images/projects/Legal Case Analysis.png", alt: "Legal Case Analysis" },
    description: "A specialized text analysis system for legal documents. The solution demonstrates how natural language processing and machine learning techniques can effectively extract critical information, accurately classify documents, and identify meaningful patterns in complex legal texts, with practical applications in legal research, regulatory compliance, and case outcome prediction.",
    keyFeatures: [
      "Legal Document Classification",
      "Information Extraction from Legal Texts",
      "Case Outcome Prediction",
      "Legal Citation Network Analysis",
      "Document Summarization",
      "Legal Language Processing"
    ],
    implementationDetails: "Created specialized text preprocessing pipelines tailored to legal documents, developed custom named entity recognition for legal entities and terminology, built accurate document classification systems for various case types, engineered information extraction algorithms for key facts and holdings, constructed citation network analysis tools for mapping precedent relationships, and designed predictive models for forecasting case outcomes based on comprehensive textual features.",
    technologies: ["Python", "Natural Language Processing", "Text Mining", "Machine Learning", "Information Extraction", "Document Classification", "Network Analysis", "Legal Informatics"]
  },
  {
    id: "dream-sentiment-analysis",
    highlight: false,
    title: "Deiphering Dreams: Dream Sentiment Analysis",
    url: "https://github.com/Gabe7430/Projects/tree/main/Deiphering%20Dreams%20-%20Dream%20Sentiment%20Analysis",
    image: { src: "/images/projects/Deciphering Dreams - Dream Sentiment Analysis.png", alt: "Deiphering Dreams: Dream Sentiment Analysis" },
    description: "An innovative analysis of sentiment and emotional content in dream narratives. The work demonstrates how natural language processing techniques can reveal hidden emotional patterns and recurring themes in dream reports, offering unique insights into the complex affective dimensions of human dreaming experiences.",
    keyFeatures: [
      "Sentiment Analysis of Dream Narratives",
      "Emotion Detection in Text",
      "Theme Identification",
      "Linguistic Feature Extraction",
      "Dream Content Classification",
      "Visualization of Emotional Patterns"
    ],
    implementationDetails: "Developed specialized text preprocessing techniques for dream narratives, created hybrid sentiment analysis using both lexicon-based and machine learning approaches, built multi-dimensional emotion detection systems (capturing joy, fear, anger, etc.), implemented theme extraction using topic modeling, and designed interactive visualization tools for exploring complex emotional patterns across diverse dreams and dreamers.",
    technologies: ["Python", "Natural Language Processing", "Sentiment Analysis", "Emotion Detection", "Text Classification", "Topic Modeling", "Psychological Text Analysis", "Data Visualization"]
  },
  {
    id: "graph-neural-network",
    highlight: false,
    title: "Graph Neural Network, Decision Tree Learning, and Clustering Data Streams",
    url: "https://github.com/Gabe7430/Projects/tree/main/Graph%20Neural%20Network%2C%20Decision%20Tree%20Learning%2C%20and%20Clustering%20Data%20Streams",
    image: { src: "/images/projects/Graph Neural Network, Decision Tree Learning, and Clustering Data Streams.png", alt: "Graph Neural Network, Decision Tree Learning, and Clustering Data Streams" },
    description: "A comprehensive suite of machine learning techniques for graph and streaming data analysis. The solution leverages graph neural networks for effective learning on complex graph-structured data, interpretable decision tree algorithms for transparent classification, and adaptive clustering methods specifically designed for data streams that continuously evolve over time.",
    keyFeatures: [
      "Graph Neural Network Implementation",
      "Decision Tree Learning Algorithms",
      "Data Stream Clustering",
      "Graph Representation Learning",
      "Incremental Learning for Streams",
      "Model Evaluation on Dynamic Data"
    ],
    implementationDetails: "Engineered state-of-the-art graph neural networks with optimized message passing and aggregation functions, developed decision tree learning algorithms with multiple splitting criteria and adaptive pruning methods, and created robust data stream clustering algorithms capable of handling concept drift and rapidly evolving patterns. Incorporated comprehensive graph preprocessing pipelines, feature engineering for structured data, and specialized evaluation metrics designed for dynamic environments.",
    technologies: ["Python", "PyTorch", "TensorFlow", "Graph Neural Networks", "Decision Trees", "Data Stream Mining", "Incremental Learning", "Graph Algorithms"]
  },
  {
    id: "kl-divergence-mnist",
    highlight: false,
    title: "KL Divergence, MNIST Image Classification, MDP, and RL",
    url: "https://github.com/Gabe7430/Projects/tree/main/KL%20Divergence%2C%20MNIST%20Image%20Classification%2C%20MDP%2C%20and%20RL",
    image: { src: "/images/projects/KL Divergence, MNIST Image Classification, MDP, and RL.png", alt: "KL Divergence, MNIST Image Classification, MDP, and RL" },
    description: "A diverse collection of machine learning and reinforcement learning implementations. The work explores practical applications of Kullback-Leibler divergence for precise distribution comparison, image classification techniques applied to the MNIST dataset, comprehensive Markov Decision Process modeling, and reinforcement learning algorithms for optimal sequential decision-making.",
    keyFeatures: [
      "KL Divergence Implementation and Applications",
      "MNIST Handwritten Digit Classification",
      "Markov Decision Process Modeling",
      "Reinforcement Learning Algorithms",
      "Distribution Comparison Techniques",
      "Policy Optimization Methods"
    ],
    implementationDetails: "Developed KL divergence calculations for precise probability distribution comparisons, built optimized convolutional neural networks for high-accuracy MNIST image classification, created comprehensive MDP formulations with well-defined states, actions, transitions, and rewards, and implemented reinforcement learning algorithms including Q-learning and policy gradient methods. Designed interactive visualization tools for monitoring model behavior and tracking learning progress across training iterations.",
    technologies: ["Python", "TensorFlow", "PyTorch", "Information Theory", "Image Classification", "Reinforcement Learning", "Markov Decision Processes", "Deep Learning"]
  }
];
