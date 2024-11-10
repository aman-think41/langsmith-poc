
## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   - Create a `.env` file in the root directory of the project.
   - Copy the contents from `.env.example` and fill in your OpenAI and LangChain credentials.

5. **Set up required files**
   - Create a `guidelines.txt` file in the root directory of the project.
   - Create a `questions.json` file in the root directory of the project.

6. **Create Chat For Testing**
   ```bash
   python chat-gen.py
   ```

6. **Run the Tests**
   ```bash
   python runner.py
   ```

### Results

The results will be saved in the langsmith dashboard.
We can also transform the results according to our needs through the SDK.