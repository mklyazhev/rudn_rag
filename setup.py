from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

    setup(
        name='llm_rag',
        version='0.1',
        packages=find_packages(),
        install_requires=[
            'torch==2.2.1',
            'transformers==4.40.2',
            'langchain==0.1.16',
            'langchain-chroma==0.1.1',
            'langchain-openai==0.1.7',
            'langchain-community==0.0.34',
            'langchain-text-splitters==0.0.2',
            'gigachat==0.1.35',
            'unstructured[md]==0.11.8',
            'sentence-transformers==2.7.0',
            'bitsandbytes==0.43.1',
            'accelerate==0.30.1',
            'xformers==0.0.25 # 0.0.25/0.0.20',
            'aiogram==3.6.0',
            'loguru==0.7.2',
            'pydantic==2.7.1',
            'pydantic-settings==2.2.1'
        ],
        author='Vasiliy',
        author_email='i@livasan.ru',
        description='LLM with RAG',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/mklyazhev/rudn_rag/tree/readme_branch',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.6',
    )