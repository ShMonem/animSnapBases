
After reproducing the data as described in the main README.md, you can test how close your bases matrices to the ones computed for the paper.
1. Testing in ``Commandline``
    - Test the computed PCA bases
        ```Commandline
        python3 test\compare_npy_files.py test\PCA_using_F_50K200.npy results\bunny\q_bases\PCA_alignedRigid_Volkwein_Standarized_Local_nonOrthogonalized_Release\50outOf50_Frames_\1_increament_200_alignedRigid_bases\using_F_50K200.npy
        ```
    - Test the SPLOCS bases:
        ```Commandline
        python3 test\compare_npy_files.py test\SPLOCS_using_F_50K200.npy results\bunny\q_bases\SPLOCS_alignedRigid_Volkwein_Standarized_Local_nonOrthogonalized_Release\50outOf50_Frames_\1_increament_200_alignedRigid_bases\using_F_50K200.npy 
        ```
    you expect somthing like
    ```
    File one contains a (200, 14290, 3) tensor, and file two (200, 14290, 3)
    checking if identical ... True
    checking if close ... True
     testing the sparsity of a
     ... not sparse.
     testing the sparsity of b
     ... not sparse.
    ```
   
2- **TODO** : Running full code and tests in Linux using a ``.sh`` 