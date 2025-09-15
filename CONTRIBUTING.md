## Contributing Guideline

0. You must have maintainer access to the repository, or make a fork of the repository.
    ```
      git clone <clone-url>
      cd impact-of-bias-in-fr
    ```

1. Make a new branch, calling it based on one of the following nametypes:
  - `feature/<what you are building>`
  - `bug-fix/<what you are fixing>`
  - `docs/<changing readme, adding paragraph about docs in XXXXX>`

2. Make your additions or changes, and then commit and push.
  ```
    git add <files>
    git commit -m "4-8 words about what is new"
    git push
  ```

3. On the GitHub webpage, create a new Merge Request (to merge your changes to the main branch).
4. In the Merge Request (also known as Pull Request (PR)), specify clearly 
  - what issue you fixed ("for issue #2 - model retrained to have less bias on class X")
  - a description of what has been changed since last time
  - how to reproduce it on another developer's computer, if others wish to test it also
5. Wait for another member to check if everything is working as expected, and that there are no conflicts.
6. Then, the merge request will be approved and changes shall be merged to the main branch.

> It is the developer's responsibility to ensure that the code changes provided are accurate to the issue being fixed, and that it does not compromise/damage already working code or modules in the repository in the last given version.
>
> > In case there are CI/CD pipelines, they must successfully complete and not be aborted. If that is done, it is not a guarantee that the code is ready/suitable to be merged (unless very few exceptions).
