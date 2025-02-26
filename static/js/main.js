// static/js/main.js
// 收藏功能相关
const getLikedStatus = (path) => {
    const likedImages = JSON.parse(localStorage.getItem('likedImages')) || {};
    return likedImages[path] || false;
};

const updateLikedStatus = (path, isLiked) => {
    const likedImages = JSON.parse(localStorage.getItem('likedImages')) || {};
    likedImages[path] = isLiked;
    localStorage.setItem('likedImages', JSON.stringify(likedImages));
};

// 图片信息展示
const showImageInfo = (path, faceScores, landmarkScores) => {
    const formatScore = (arr) => {
        if (!arr || arr.length === 0) return '无可用数据';
        const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
        return avg.toFixed(4) + ` (${arr.length}个检测结果)`;
    };

    document.getElementById('infoPath').textContent = path;
    document.getElementById('infoFaceScore').textContent = formatScore(faceScores);
    document.getElementById('infoLandmarkScore').textContent = formatScore(landmarkScores);
    document.getElementById('infoModal').style.display = 'flex';

    const isLiked = getLikedStatus(path);
    const likeButton = document.getElementById('likeButton');
    likeButton.textContent = isLiked ? '取消收藏' : '收藏';
    likeButton.onclick = () => likeImage(path);
};

// 模态框控制
const closeModal = () => {
    document.getElementById('infoModal').style.display = 'none';
};

window.onclick = (event) => {
    const modal = document.getElementById('infoModal');
    if (event.target === modal) closeModal();
};

// 收藏操作
const likeImage = (path) => {
    const button = document.getElementById('likeButton');
    const isLiked = getLikedStatus(path);
    const action = isLiked ? 'unlike' : 'like';

    fetch('/like_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            path: path,
            action: action
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const newIsLiked = data.action === 'like';
            updateLikedStatus(path, newIsLiked);
            button.textContent = newIsLiked ? '取消收藏' : '收藏';
            alert(data.action === 'like' ? '收藏成功' : '已取消收藏');
        } else {
            alert('操作失败');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('操作失败');
    });
};