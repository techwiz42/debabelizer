# GitHub Account Migration Guide

This document provides step-by-step instructions for migrating your GitHub account from `techwiz42` to `cyberiad-ai` and updating your email address.

## Overview

You have three options for transitioning to your new GitHub identity:

1. **Rename Current Account** ⭐ (Recommended)
2. **Create New Account + Transfer Repos** ⭐⭐⭐
3. **Create New Account + Fork/Clone** ⭐⭐⭐⭐ (Not recommended)

## Option 1: Rename Current Account (Recommended)

**Difficulty**: Very Easy  
**Time**: 5-10 minutes  
**Benefits**: Preserves all history, stars, followers, and contribution graph

### Step 1: Change Email Address

1. **Go to GitHub Settings**
   - Navigate to [github.com/settings/emails](https://github.com/settings/emails)

2. **Add New Email**
   - Click "Add email address"
   - Enter your new email (e.g., `contact@cyberiad.ai`)
   - Click "Add"

3. **Verify New Email**
   - Check your email for verification link
   - Click the verification link

4. **Set as Primary**
   - Return to Email settings
   - Select your new email as primary
   - Click "Save"

5. **Optional: Remove Old Email**
   - You can keep the old email as backup
   - Or remove it once everything is working

### Step 2: Change Username

1. **Go to Account Settings**
   - Navigate to [github.com/settings/admin](https://github.com/settings/admin)

2. **Change Username**
   - Click "Change username"
   - Enter new username: `cyberiad-ai`
   - Read the warnings and confirm
   - Click "I understand, let's change my username"

3. **Automatic Benefits**
   - All URLs automatically redirect: `techwiz42` → `cyberiad-ai`
   - All repositories, issues, PRs maintain their links
   - GitHub handles the redirection seamlessly

### Step 3: Update Local Git Configuration

```bash
# Update global git config for future commits
git config --global user.email "contact@cyberiad.ai"
git config --global user.name "Cyberiad AI"

# Update remote URLs for existing repositories
find . -name ".git" -type d -exec sh -c '
    cd "{}" && cd .. && 
    old_url=$(git remote get-url origin)
    new_url=$(echo "$old_url" | sed "s/techwiz42/cyberiad-ai/g")
    git remote set-url origin "$new_url"
    echo "Updated $(pwd): $old_url -> $new_url"
' \;
```

### Step 4: Update Project Files

For each of your projects, update references to your old username:

```bash
# Update package.json files
find . -name "package.json" -exec sed -i 's/techwiz42/cyberiad-ai/g' {} \;

# Update README.md files  
find . -name "README.md" -exec sed -i 's/techwiz42/cyberiad-ai/g' {} \;

# Update any hardcoded GitHub URLs in documentation
grep -r "github.com/techwiz42" . --include="*.md" --include="*.rst" --include="*.txt"
```

## Option 2: Create New Account + Transfer Repos

**Difficulty**: Moderate  
**Time**: 1-2 hours  
**Use Case**: If you want to selectively migrate only certain projects

### Step 1: Create New Account

1. **Sign out** of your current GitHub account
2. **Create new account** at [github.com/join](https://github.com/join)
   - Username: `cyberiad-ai`
   - Email: Your new email address
   - Verify the account

### Step 2: Transfer Repositories

For each repository you want to transfer:

1. **Go to repository settings**
   - Navigate to the repo → Settings → General
   - Scroll to "Danger Zone"

2. **Transfer ownership**
   - Click "Transfer"
   - Enter new owner: `cyberiad-ai`
   - Confirm transfer

3. **Accept transfer**
   - Switch to new account
   - Accept the transfer request

### Step 3: Update Integrations

After transferring repositories:

- **CI/CD Systems**: Update webhooks in Travis CI, CircleCI, etc.
- **Package Registries**: Update NPM, PyPI package ownership
- **Domain/DNS**: Update any GitHub Pages custom domains
- **Third-party Apps**: Reconnect apps that integrate with GitHub

## Post-Migration Checklist

### Immediate Updates Required

- [ ] **Update git remotes** in local repositories
- [ ] **Update CI/CD configurations** (GitHub Actions, etc.)
- [ ] **Update package.json** repository URLs
- [ ] **Update README badges** and links
- [ ] **Update documentation** with new GitHub URLs
- [ ] **Notify collaborators** of the username change

### Services to Update

- [ ] **NPM packages**: `npm adduser` with new account
- [ ] **Docker Hub**: Update linked GitHub account
- [ ] **Personal website**: Update GitHub profile links
- [ ] **Social media**: Update GitHub profile links in bio
- [ ] **Email signatures**: Update GitHub profile links
- [ ] **Resume/CV**: Update GitHub username

### Optional Git History Updates

If you want to update commit email history (advanced):

```bash
# WARNING: This rewrites git history - only do this if necessary
git filter-branch --env-filter '
OLD_EMAIL="old@email.com"
CORRECT_NAME="Cyberiad AI"
CORRECT_EMAIL="contact@cyberiad.ai"

if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags

# Force push to update remote history (use with caution!)
git push --force --tags origin 'refs/heads/*'
```

## Username Requirements

### Valid Usernames
- `cyberiad-ai` ✅ (hyphens allowed)
- `cyberiadai` ✅ (no special characters)
- `cyberiad` ✅ (if available)

### Invalid Usernames
- `cyberiad_ai` ❌ (underscores display as hyphens)
- `cyberiad.ai` ❌ (dots not allowed)
- `cyberiad/ai` ❌ (slashes not allowed)

## Important Notes

### What Gets Preserved (Rename Option)
- ✅ Contribution graph and activity
- ✅ Stars, forks, and watchers
- ✅ Issues and pull requests
- ✅ Repository history and releases
- ✅ Gists and comments
- ✅ Followers and following
- ✅ Organization memberships

### What Needs Manual Updates
- ⚠️ Local git remote URLs
- ⚠️ CI/CD webhook URLs
- ⚠️ Package registry configurations
- ⚠️ Third-party integrations
- ⚠️ Documentation and README files
- ⚠️ Personal website links

### GitHub's Automatic Redirects
- Old URLs redirect for a reasonable time period
- Clone operations work with old URLs initially
- Search results update gradually
- External sites may cache old URLs

## Verification Steps

After completing the migration:

1. **Test repository access**:
   ```bash
   git clone https://github.com/cyberiad-ai/debabelizer.git
   ```

2. **Verify redirects work**:
   ```bash
   curl -I https://github.com/techwiz42/debabelizer
   # Should return 301 redirect to cyberiad-ai
   ```

3. **Check profile page**:
   - Visit `https://github.com/cyberiad-ai`
   - Verify all repositories are visible
   - Check contribution graph

4. **Test integrations**:
   - Trigger a CI/CD build
   - Verify webhooks are working
   - Test any automated deployments

## Troubleshooting

### Common Issues

**Issue**: Git push fails after rename  
**Solution**: Update remote URL:
```bash
git remote set-url origin https://github.com/cyberiad-ai/repo-name.git
```

**Issue**: CI/CD builds failing  
**Solution**: Update webhook URLs in CI service settings

**Issue**: Package registry shows old username  
**Solution**: Re-authenticate: `npm adduser` or equivalent

**Issue**: Third-party apps can't access repos  
**Solution**: Revoke and re-authorize app permissions

### Getting Help

- **GitHub Support**: [support.github.com](https://support.github.com)
- **Documentation**: [docs.github.com](https://docs.github.com)
- **Community**: [github.community](https://github.community)

## Timeline Recommendation

**Day 1**: Change email and username  
**Day 2-3**: Update local repositories and essential integrations  
**Week 1**: Update documentation and notify collaborators  
**Week 2**: Update external services and profiles  

This gradual approach ensures you don't break anything critical while transitioning to your new GitHub identity.

## Success Metrics

You'll know the migration is complete when:
- [ ] All repositories accessible via new username
- [ ] Local git operations work normally
- [ ] CI/CD systems building successfully
- [ ] No broken links in documentation
- [ ] External services recognize new username
- [ ] Search results show new username

**Total estimated time for rename option**: 2-4 hours spread over a week  
**Total estimated time for transfer option**: 4-8 hours over several days