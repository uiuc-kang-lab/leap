from setuptools import setup, find_packages

setup(
    name='autopipeline',
    version='0.1.393',
    packages=find_packages(),
    package_data={
        # Include files from the "data" directory in the "autopipeline" package
        'autopipeline': ['data/**/*'],
    },
    include_package_data=True,
    license='LICENSE',
    install_requires=[
        'tiktoken',
        'IPython',
        'graphviz',
        'openai',
        'pandas',
        'PyMuPDF',
        'pytesseract',
        'Pillow',
        'gensim',
        'nltk',
        'textblob',
        'scikit-learn',
        'pandasql',
        'spacy',
        'requests',
        'huggingface_hub'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
