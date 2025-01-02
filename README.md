# Event Recommendation System

A simple neural network-based recommendation system that suggests events to users based on their search history and event characteristics. This project demonstrates fundamental concepts in recommendation systems and neural networks, making it a valuable addition to a junior data scientist's portfolio.

## Project Overview

This system implements a content-based filtering approach using a simple neural network to recommend events to users. The system learns from user search history and event attributes to predict which future events might interest them.

### Key Features

- Content-based filtering using neural network
- Dynamic category encoding
- Date normalization
- Personalized event recommendations
- Simple but effective architecture for learning purposes

## Technical Implementation

### Data Structure

The system uses three main classes:

1. **Event Class**
   - Stores event details (name, category, date)
   - Implements feature engineering:
     - Dynamic one-hot encoding for categories
     - Date normalization to [0,1] range
     - Feature vector generation

2. **User Class**
   - Maintains user ID and search history
   - Tracks user interactions with events

3. **SimpleNeuralNet Class**
   - Implements a single-layer neural network
   - Features:
     - Sigmoid activation function
     - Gradient descent optimization
     - Binary classification (user interest prediction)

### Machine Learning Approach

The system employs:
- Binary classification for user preferences
- Gradient descent for model optimization
- Sigmoid activation for probability outputs
- Feature engineering with one-hot encoding and date normalization

## Usage Example

```csharp
// Create sample events
Event[] events = new Event[]
{
    new Event("Community Meeting", "Community", new DateTime(2024, 12, 10)),
    new Event("Music Concert", "Music", new DateTime(2024, 11, 22)),
    // ... more events
};

// Initialize user and add search history
User user = new User(1);
user.AddSearch(events[0]);
user.AddSearch(events[2]);

// Train the model
SimpleNeuralNet model = new SimpleNeuralNet(learningRate: 0.01);
model.Train(user, events, categoryMap, epochs: 1000);

// Get recommendations
List<Event> recommendedEvents = model.RecommendEvents(user, events, categoryMap, topN: 3);
```

## Technical Considerations

### Strengths
- Simple and interpretable architecture
- Efficient training process
- Dynamic category handling
- Normalized date features
- Extensible design

### Limitations
- Single-layer architecture
- Binary classification approach
- Limited feature engineering
- No cold start handling
- Single user focus

## Future Improvements

1. **Architecture Enhancements**
   - Implement multi-layer neural network
   - Add more sophisticated feature engineering
   - Include collaborative filtering elements

2. **Feature Additions**
   - User embedding layers
   - Multiple user support
   - Cold start handling
   - Cross-validation implementation

3. **Performance Optimization**
   - Batch processing
   - Learning rate scheduling
   - Regularization techniques
   - Model persistence

## Learning Outcomes

This project demonstrates proficiency in:
- Neural network implementation
- Feature engineering
- Recommendation systems
- C# programming
- Object-oriented design
- Machine learning fundamentals

## Installation and Setup

1. Clone the repository
2. Ensure .NET Core SDK is installed
3. Build the project:
   ```bash
   dotnet build
   ```
4. Run the project:
   ```bash
   dotnet run
   ```

## Dependencies

- .NET Core 3.1 or higher
- System.Linq
- System.Collections.Generic

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
