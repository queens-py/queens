0. Onboarding
=============
Welcome to the QUEENS project! 👑

We are happy to have you here and created this onboarding guide to help you get started with the
project, especially if you are new to open-source development.

Steps to get started with QUEENS
--------------------------------

#. Clone the QUEENS repository to your local machine via SSH:

   .. code-block:: bash

      git clone git@github.com:queens-py/queens.git

#. Read through our `introduction <https://queens-py.github.io/queens/intro.html>`_ and
   follow the instructions in the "Installation" section to set up your local environment.

#. If you plan to contribute to the project, please read through our
   `contributing guidelines <https://queens-py.github.io/queens/contributing.html>`_ and
   follow the instructions there to configure our git-hooks.

#. Install an integrated development environment (IDE) of your choice if you do not already have
   one.
   We recommend using `PyCharm <https://www.jetbrains.com/pycharm/>`_ or
   `VS Code <https://code.visualstudio.com/download>`_ with some
   `additional extensions <https://thedeveloperspace.com/10-essential-vs-code-extensions-for-python-
   development-in-2024/>`_.

   *Optional*: For VS Code, we further recommend adding the following settings to your
   :code:`~/.config/Code/User/settings.json` after finding out the path to your QUEENS environment
   via :code:`conda info --envs`:

   .. code-block:: json

      "python.defaultInterpreterPath": "<path_to_your_QUEENS_environment>/bin/python",
      "editor.formatOnSave": true,
      "editor.defaultFormatter": "ms-python.black-formatter",
      "black-formatter.args": ["--line-length", "100"],

#. Browse through our `documentation <https://queens-py.github.io/queens/overview.html>`_ and our
   other tutorials to get a first impression.

#. *Optional*: If you are comfortable with it, feel free to introduce yourself in our
   `discussions forum <https://github.com/queens-py/queens/discussions/categories/introduce-
   yourself>`_.
   For example, you could share your name, prior experience with coding, and a high level
   description of what you will be working on.

#. Last but not least, we ask you to follow our
   `code of conduct <https://github.com/queens-py/queens/blob/main/CODE_OF_CONDUCT.md>`_ so we can
   all thrive in the Qommunity 💂‍♀️👑💂‍♂️

.. note::
   If you find that any of the information in this onboarding guide is outdated or incorrect, please
   let us know by creating an issue or a pull request on
   `GitHub <https://github.com/queens-py/queens/issues>`_.


Additional information for rookies
----------------------------------

Git
***

We use Git for tracking changes in our files.
This way, we can work simultaneously on different parts of the code without disturbing others.
For a general introduction to Git, you can check out
`this article <https://www.freecodecamp.org/news/what-is-git-and-how-to-use-it-c341b049ae61/>`_ or
`this video <https://www.youtube.com/watch?v=8JJ101D3knE>`_.
However, the simplest way to learn Git is by using it.


Python
******

If you have made it this far, you will probably end up coding in Python.
For an introduction to Python, you can check out
`this written tutorial <https://docs.python.org/3/tutorial/>`_ or
`this video <https://www.youtube.com/watch?v=kqtD5dpn9C8>`_.

Linux
*****

QUEENS is currently developed on Linux so we advise you to do the same.
Do not worry if you are not familiar with Linux.
There is no big difference from Mac or Windows besides the extended use of the terminal, also known
as the command line.
At first, this might seem cumbersome, but once you master a basic set of commands, it becomes super
efficient!
Check out `this article <https://maker.pro/linux/tutorial/basic-linux-commands-for-beginners>`_ to
start learning some basic Linux commands.
*Bonus:* If you really want to become a nerd, also have a look at the
`VIM editor <https://opensource.com/article/19/3/getting-started-vim>`_.
