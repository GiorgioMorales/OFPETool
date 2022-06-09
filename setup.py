import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='OFPETool',
    version='0.0.1',
    author='Giorgio Morales - Montana State University',
    author_email='giorgiol.moralesluna@student.montana.edu',
    description='ML for Prediction in Precision Agriculture',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/GiorgioMorales/OFPETool',
    project_urls={"Bug Tracker": "https://github.com/GiorgioMorales/OFPETool/issues"},
    license='MIT',
    packages=['AdaBoost', 'Predictor', 'PredictorStrategy', 'pytorch_ssim'],
    install_requires=['requests'],
)