from setuptools import setup

setup(
    name="tali",
    version="3.4.3.1",
    packages=[
        "tali.datasets",
        "tali.models",
        "tali.models.auto_builder",
        "tali.runner",
        "tali.utils",
        "tali.base.callbacks",
        "tali.base.utils",
        "tali.base.vendor",
        "tali.base",
        "tali",
    ],
    url="",
    license="GNU General Public License v3.0",
    author="Antreas Antoniou",
    author_email="a.antoniou@ed.ac.uk",
    description="TALI - A multi modal dataset consisting of Temporally correlated Audio, Images (including Video) and Language",
)
