from setuptools import setup, find_packages

setup(
    name='rolling-shutter-image-as-an-imu',
    version='0.1',
    packages=find_packages(),
    # entry_points={
    #     'console_scripts': [
    #         'data_cleaner=project_final.data_cleaning.data_cleaner:main',
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)