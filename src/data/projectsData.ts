import PoseEstimationImage from './img/6d-pose-estimation/rgb_sampling.png';
import EnsembleRLImage from './img/ensemble-rl/ensemble_results.png';
import EightballImage from './img/eightball-protocol/Eightball_Protocol.jpeg';
import SmartTrafficImage from './img/ai-traffic-light/3D_simulation.png';
import LegalCaseImage from './img/llm-court-rulings/Analysis_Pipeline.png'
import DreamImage from './img/dream-analysis/dream.png'
import MovieRecommendationImage from './img/chatbot/chatbot.png'

export const projects = [
    {
      id: "pose-estimation",
      highlight: true,
      image: PoseEstimationImage, 
      title: "Enhanced PoseCNN for Accurate 6D Pose Estimation",
      description:
        "Improved PoseCNN architecture for precise 6D pose estimation in robotic applications, particularly in complex object manipulation scenarios.",
      shortDescription: "Improved PoseCNN architecture for precise 6D pose estimation in robotic applications.",
      achievements: [
        "Generated a comprehensive 2GB synthetic dataset featuring 25 YCB objects",
        "Created diverse camera angles and occlusion scenarios to simulate real-world conditions",
        "Extended PoseCNN (VGG16-based) architecture for improved pose estimation",
        "Developed sophisticated data generation pipeline for training scenarios",
      ],
      technologies: ["PyTorch", "AWS", "Computer Vision Libraries", "VGG16", "PoseCNN"],
      skills: ["Deep Learning", "Computer Vision", "Dataset Generation", "Robotics"],
    },
    {
      id: "ensemble-rl",
      highlight: true,
      image: EnsembleRLImage, 
      title: "Ensemble RL for Portfolio Optimization",
      description:
        "Developed a stacking-based reinforcement learning strategy for portfolio optimization, combining five advanced RL algorithms (A2C, DDPG, PPO, TD3, SAC) to enhance trading performance.",
      shortDescription: "Stacking-based reinforcement learning strategy for portfolio optimization using five advanced RL algorithms.",
      achievements: [
        "Achieved higher average portfolio returns compared to traditional single-agent approaches",
        "Reduced variance in trading performance through ensemble methodology",
        "Successfully integrated diverse RL frameworks including Actor-Critic and Policy Gradient methods",
      ],
      technologies: ["Python", "PyTorch", "Multiple RL Frameworks", "Financial Analysis Tools"],
      skills: ["Reinforcement Learning", "Financial Engineering", "Ensemble Methods", "Algorithm Design"],
    },
    {
      id: "eightball-protocol",
      highlight: true,
      image: EightballImage,
      title: "EightBall Protocol - Prediction Market Platform",
      description: "Developed a prediction market system utilizing a customized Constant Function Market Maker (CFMM) for efficient liquidity management. The protocol enables users to provide liquidity, make predictions, and resolve outcomes in a decentralized manner.",
      shortDescription: "Prediction market system utilizing a customized CFMM for efficient liquidity management.",
      achievements: [
        "Designed and implemented an Automated Market Maker (AMM) that allows market initialization at any probability",
        "Optimized liquidity injection mechanisms to eliminate leftover shares",
        "Developed comprehensive testing infrastructure to simulate various market scenarios using Foundry",
        "Modified Uniswap's core mechanisms and introduced new mathematical models for dynamic probability balancing"
      ],
      technologies: ["Solidity", "Ethereum", "Smart Contracts", "CFMMs", "Foundry (Testing Framework)"],
      skills: ["Smart Contract Development", "DeFi Protocol Design", "Mathematical Modeling", "Test-Driven Development"],
    },
    {
      id: "smart-traffic-light",
      highlight: true,
      image: SmartTrafficImage,
      title: "AI-Powered Smart Traffic Light",
      description: "Developed a traffic light system using computer vision and reinforcement learning to reduce congestion. Leveraged YOLO for real-time vehicle and pedestrian detection and trained a deep Q-learning model to optimize light changes based on traffic flow. Designed to replace inefficient timed lights and limited pressure-sensor systems, improving urban traffic management.",
      shortDescription: "Traffic light system using computer vision and reinforcement learning to reduce congestion.",
      achievements: [
        "Utilized YOLOv3 for real-time object detection create features to input into the RL model",
        "Trained a Deep Q-learning model to dynamically optimize traffic light timings",
        "Integrated the SUMO traffic simulator to evaluate traffic flow efficiency under various conditions",
        "Achieved significant reductions in average vehicle wait times compared to traditional fixed-timer lights"
      ],
      technologies: ["YOLO", "Deep Q-Learning", "Computer Vision",  "SUMO"],
      skills: ["Machine Learning", "Computer Vision", "Reinforcement Learning", "Deep Learning", "Smart Infrastructure", "Traffic Optimization"],
    },
    {
      id: "legal-case-analysis",
      highlight: false,
      image: LegalCaseImage,
      title: "Legal Case Analysis Using Advanced ML Techniques",
      description: "Developed a sophisticated legal case analysis system using Legal BERT and multiple clustering algorithms to organize and analyze court cases effectively.",
      shortDescription: "A legal case analysis system using Legal BERT and clustering algorithms to analyze court cases effectively.",
      achievements: [
        "Implemented Legal BERT for specialized legal document processing and embeddings",
        "Designed and compared five different clustering approaches:",
        "K-means and K-means++ for basic case categorization",
        "Expectation-Maximization (EM) for soft clustering of overlapping legal categories",
        "DBSCAN for density-based automatic cluster detection",
        "Hierarchical clustering with Ward's method for exploratory analysis",
        "Created a comprehensive preprocessing pipeline for legal documents",
        "Developed methods to handle complex relationships between related cases"
      ],
      technologies: ["Legal BERT", "AWS", "Python", "Scikit-learn", "Transformer Models"],
      skills: ["Natural Language Processing", "Clustering Algorithms", "Legal Document Analysis", "Machine Learning"]
    },
    {
      id: "dream-analysis",
      highlight: false,
      image: DreamImage,
      title: "Sentiment Analysis of Dream Journals",
      description: "This project explores the use of large language models (LLMs), specifically RoBERTa, to automate sentiment analysis of dream reports, leveraging the Sleep and Dream Database. The goal is to assess whether LLMs can accurately classify emotions in dream narratives and identify recurring emotional themes, with a comparison against the Hall and Van de Castle (HVDC) framework and traditional sentiment analysis tools like VADER.",
      shortDescription: "Sentiment Analysis of Dream Journals using RoBERTa to automate the classification of emotions in dream reports.",
      achievements: [
        "Implemented RoBERTa for sentiment analysis of dream reports",
        "Fine-tuned RoBERTa for multi-label classification of emotions in dreams using Binary Cross-Entropy Loss",
        "Evaluated model performance using F1-score, Precision, Recall, and AUROC",
      ],
      technologies: ["RoBERTa", "GCP", "Python", "Scikit-learn", "Transformer Models"],
      skills: ["Natural Language Processing", "Clustering Algorithms", "Dream Report Analysis", "Machine Learning", "Deep Learning"]
    },
    {
      id: "movie-recommendation-chatbot",
      highlight: false,
      image: MovieRecommendationImage,
      title: "Chatbot for Movie Recommendations",
      description: "This project involves building a chatbot inspired by ELIZA, designed to recommend movies to users. The project progresses from a basic single-agent version to more advanced features, incorporating Large Language Models (LLMs) for enhanced interaction. The chatbot initially functions in a 'starter' mode, providing movie recommendations based on user input, and later transitions to LLM prompting and programming modes for more complex interactions.",
      shortDescription: "Develop a chatbot for movie recommendations, evolving from a basic model to utilizing LLMs for advanced features.",
      achievements: [
        "Implemented a movie recommendation system using collaborative filtering.",
        "Integrated LLMs for enhanced chatbot interactions.",
        "Developed a multi-mode chatbot with starter, LLM prompting, and LLM programming capabilities."
      ],
      technologies: ["Python", "OpenAI API", "Collaborative Filtering", "LLMs"],
      skills: ["Natural Language Processing", "Chatbot Development", "Machine Learning", "Collaborative Filtering"]
    }
  ];