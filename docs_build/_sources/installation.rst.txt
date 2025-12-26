Installation
============

Prerequisites
-------------
*   **OS**: Windows, Linux, or macOS
*   **Python**: 3.8 or higher
*   **C++ Build Tools**: Required for `gymnasium[box2d]` (e.g., Visual Studio Build Tools on Windows).

Step-by-Step Installation
-------------------------

1.  **Clone the Repository**

    .. code-block:: bash

        git clone https://github.com/yourusername/PC-UDRL.git
        cd PC-UDRL

2.  **Create a Virtual Environment** (Recommended)

    .. code-block:: bash

        # Windows
        python -m venv venv
        .\venv\Scripts\activate

        # Linux/Mac
        python3 -m venv venv
        source venv/bin/activate

3.  **Install Dependencies**

    This project requires PyTorch, Gymnasium (with Box2D support), and d3rlpy for baselines.

    .. code-block:: bash

        pip install -r requirements.txt
        # Or manually:
        pip install torch gymnasium[box2d] d3rlpy pandas matplotlib seaborn h5py shimmy swig

4.  **Install the Package in Editable Mode**

    .. code-block:: bash

        pip install -e .

Troubleshooting
---------------

*   **Box2D Error**: If you encounter errors installing `gymnasium[box2d]`, ensure `swig` is installed.
    *   *Windows*: `pip install swig`
*   **d3rlpy Compatibility**: Use d3rlpy v2.x or v3.x compatible with your PyTorch version.
