namespace Neural_Network_for_One_User
{
    internal class Program
    {
        static void Main(string[] args)
        {


         
        
            // Creating a dynamic map for category encoding
            Dictionary<string, int> categoryMap = new Dictionary<string, int>();

            // Creating sample events
            Event[] events = new Event[]
            {
            new Event("Community Meeting", "Community", new DateTime(2024, 12, 10)),
            new Event("Music Concert", "Music", new DateTime(2024, 11, 22)),
            new Event("Soccer Tournament", "Sports", new DateTime(2024, 10, 15)),
            new Event("Art Exhibition", "Art", new DateTime(2024, 12, 01)),
            new Event("Tech Conference", "Technology", new DateTime(2024, 9, 5)),
            new Event("Food Festival", "Food", new DateTime(2024, 8, 25))
            };

            // Creating a sample user and search history
            User user = new User(1);
            user.AddSearch(events[0]); // User searched for "Community Meeting"
            user.AddSearch(events[2]); // User searched for "Soccer Tournament"

            // Create and train the model
            SimpleNeuralNet model = new SimpleNeuralNet(learningRate: 0.01);
            model.Train(user, events, categoryMap, epochs: 1000);

            // Recommend top 3 events
            List<Event> recommendedEvents = model.RecommendEvents(user, events, categoryMap, topN: 3);

            // Display recommended events
            Console.WriteLine("Top 3 Recommended Events:");
            foreach (var recommendedEvent in recommendedEvents)
            {
                Console.WriteLine($"- {recommendedEvent.Name} ({recommendedEvent.Category}) on {recommendedEvent.Date.ToShortDateString()}");
            }


        }

        // Event class definition with dynamic one-hot encoding
        public class Event
        {
            public string Name { get; set; }
            public string Category { get; set; }
            public DateTime Date { get; set; }

            public Event(string name, string category, DateTime date)
            {
                Name = name;
                Category = category;
                Date = date;
            }

            // Convert event to a feature vector (with dynamic one-hot encoding)
            public double[] ToFeatureVector(Dictionary<string, int> categoryMap)
            {
                double[] categoryEncoding = CategoryToOneHot(Category, categoryMap);
                double normalizedDate = NormalizeDate(Date);

                // Combine category encoding and date as a feature vector
                return categoryEncoding.Concat(new double[] { normalizedDate }).ToArray();
            }

            // Dynamically generate one-hot encoding for categories
            private double[] CategoryToOneHot(string category, Dictionary<string, int> categoryMap)
            {
                // Assign an index to the category if it is new
                if (!categoryMap.ContainsKey(category.ToLower()))
                {
                    categoryMap[category.ToLower()] = categoryMap.Count;
                }

                // Create a one-hot vector of the current size of categoryMap
                double[] oneHotVector = new double[categoryMap.Count];
                int categoryIndex = categoryMap[category.ToLower()];

                // Set the category index to 1 in the one-hot vector
                oneHotVector[categoryIndex] = 1;

                return oneHotVector;
            }

            // Normalize date to a value between 0 and 1 (based on the year 2024)
            private double NormalizeDate(DateTime date)
            {
                DateTime start = new DateTime(2024, 1, 1);
                DateTime end = new DateTime(2024, 12, 31);
                double range = (end - start).TotalDays;
                return (date - start).TotalDays / range;
            }
        }

        // User class for storing search history
        public class User
        {
            public int UserId { get; set; }
            public List<Event> SearchHistory { get; set; }

            public User(int userId)
            {
                UserId = userId;
                SearchHistory = new List<Event>();
            }

            public void AddSearch(Event eventItem)
            {
                SearchHistory.Add(eventItem);
            }
        }

        // Simple Neural Network for content-based filtering (for one user)
        public class SimpleNeuralNet
        {
            private double[] weights;
            private double bias;
            private double learningRate;

            public SimpleNeuralNet(double learningRate = 0.01)
            {
                this.learningRate = learningRate;
            }

            // Initialize weights based on input size
            private void InitializeWeights(int inputSize)
            {
                weights = new double[inputSize];
                bias = 0;
                Random rand = new Random();
                for (int i = 0; i < inputSize; i++)
                {
                    weights[i] = rand.NextDouble() - 0.5; // Initialize with small random values
                }
            }

            // Train the model using search history (positive) and random events (negative)
            public void Train(User user, Event[] allEvents, Dictionary<string, int> categoryMap, int epochs)
            {
                // Ensure the weights are initialized based on the input size (categoryMap size + 1 for date)
                int inputSize = categoryMap.Count + 1;
                InitializeWeights(inputSize);

                // Convert events to feature vectors
                List<double[]> positiveSamples = user.SearchHistory.Select(e => e.ToFeatureVector(categoryMap)).ToList();
                List<double[]> negativeSamples = allEvents.Except(user.SearchHistory).Select(e => e.ToFeatureVector(categoryMap)).Take(positiveSamples.Count).ToList();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    foreach (var sample in positiveSamples)
                    {
                        TrainSample(sample, 1); // Positive sample should have output = 1
                    }

                    foreach (var sample in negativeSamples)
                    {
                        TrainSample(sample, 0); // Negative sample should have output = 0
                    }
                }
            }

            private void TrainSample(double[] sample, double target)
            {
                double output = Predict(sample);
                double error = target - output;

                // Gradient Descent Update
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] += learningRate * error * sample[i];
                }
                bias += learningRate * error;
            }

            // Predict if the user will like an event (based on a sigmoid activation function)
            public double Predict(double[] featureVector)
            {
                double z = bias;

                // Weighted sum of features
                for (int i = 0; i < weights.Length; i++)
                {
                    z += weights[i] * featureVector[i];
                }

                return Sigmoid(z);
            }

            private double Sigmoid(double z)
            {
                return 1.0 / (1.0 + Math.Exp(-z));
            }

            // Recommend events based on prediction scores
            public List<Event> RecommendEvents(User user, Event[] allEvents, Dictionary<string, int> categoryMap, int topN = 3)
            {
                // Ensure the weights are initialized based on the input size (categoryMap size + 1 for date)
                int inputSize = categoryMap.Count + 1;
                InitializeWeights(inputSize);

                // Generate scores for all events
                var eventScores = new List<Tuple<Event, double>>();
                foreach (var eventItem in allEvents)
                {
                    double[] featureVector = eventItem.ToFeatureVector(categoryMap);
                    double score = Predict(featureVector);
                    eventScores.Add(new Tuple<Event, double>(eventItem, score));
                }

                // Sort by score in descending order and take the top N events
                var recommendedEvents = eventScores.OrderByDescending(e => e.Item2).Take(topN).Select(e => e.Item1).ToList();
                return recommendedEvents;
            }
        }


    }



}
