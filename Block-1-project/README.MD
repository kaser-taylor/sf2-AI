Write a simple Python script implementing a Q-learning algorithm to solve the CartPole problem in OpenAI Gym.

So this is where I am gonna take notes on how i implement this 

p1 can you create a list of resources and things i may need to know for this project Write a simple Python script implementing a Q-learning algorithm to solve the CartPole problem in OpenAI Gym.

a1 
    -links not actually linked will post them as I read them. 
    OpenAI Gym Documentation
        - https://gymnasium.farama.org/ 

    Q-Learning Explained by OpenAI Spinning Up

    📹 Videos & Tutorials
    Q-learning from Scratch – Python Tutorial by Tech With Tim

    Reinforcement Learning Crash Course – by deeplizard

    💡 Example Notebooks
    CartPole Q-learning (Google Colab)

    Simple Q-learning on CartPole (GitHub)

How to set up a python v environment
        - Yup, you read that **perfectly** — and let me give you the tea on what’s going on 

        ---

        ###  What’s Happening
        You're on **macOS**, and the version of Python installed via **Homebrew** (or the system one) is now protected — it **won’t let you install packages system-wide** with `pip` to avoid breaking things.

         This is thanks to **[PEP 668](https://peps.python.org/pep-0668/)**, which basically says:
        > "Use a virtual environment, bestie."

        ---

        ###  What You Should Do

        You're right — it’s totally okay (and actually ideal) to make a **virtual environment per subdirectory/project**.

        Here’s the **official recipe**:

        ---

        ###  Set Up a Virtual Environment (per project)

        1. **Navigate to your project folder**  
        ```bash
        cd ~/path/to/your/project-folder
        ```

        2. **Create a virtual environment**  
        ```bash
        python3 -m venv venv
        ```

        3. **Activate it**  
        ```bash
        source venv/bin/activate  # On Mac or Linux
        ```

        *(If you see `(venv)` at the start of your terminal prompt, you’re golden.)*

        4. **Install packages into the venv**  
        ```bash
        python3 -m pip install gym numpy matplotlib
        ```

        ---

        ###  Want to Keep It Clean?

        You can also add `venv/` to your `.gitignore` so you don’t accidentally commit the virtual environment.

        ```bash
        echo "venv/" >> .gitignore
        ```

        ---

        Let me know if you want to make this smoother with an alias or script! Once you get the flow down, it’s muscle memory 
    
    - I found this helpful cause I haven't selected an interpreter before
        -  Step-by-Step: Set the Python Interpreter in VS Code
        This tells VS Code which version of Python (and which virtual environment) it should use for running, linting, and IntelliSense.

         The Shortcut
        To open the Command Palette, press:

        On macOS:
        plaintext
        Copy
        Edit
        ⇧ + ⌘ + P
        (Shift + Command + P)
        On Windows/Linux:
        plaintext
        Copy
        Edit
        Ctrl + Shift + P
         Now Select the Python Interpreter
        Once the Command Palette is open:

        Type:

        vbnet
        Copy
        Edit
        Python: Select Interpreter
        Hit Enter on that option.

        You'll see a list of Python interpreters — it might look something like this:

        ruby
        Copy
        Edit
        Python 3.12.1  (~/my-project/venv)
        Python 3.11.6  (/opt/homebrew/bin/python3)
        Python 3.9.13  (/usr/local/bin/python3)
        Pick the one inside your project, like:

        bash
        Copy
        Edit
        ./venv/bin/python
        or

        swift
        Copy
        Edit
        /Users/you/my-project/venv/bin/python
        This will make Pylance, linting, and terminal execution all work inside that venv 

         Pro Tips
        If you don't see your venv, try activating it first:

        bash
        Copy
        Edit
        source venv/bin/activate
        Then reload VS Code:
        ⇧⌘P → “Reload Window”

        You can check your current interpreter at the bottom-left of VS Code, usually something like:

        nginx
        Copy
        Edit
        Python 3.12.1 ('venv': venv)
        Clicking that does the same thing as using the Command Palette.

        Let me know if you want a screenshot guide or wanna make this the default interpreter across multiple projects 