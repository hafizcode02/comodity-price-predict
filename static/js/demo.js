$(function() {
    "use strict";
    initSparkline();
    initCounters();
    skinChanger();
    FontSetting();
    CustomPageJS();
});

// Sparkline
function initSparkline() {
    $(".sparkline").each(function() {
        var $this = $(this);
        $this.sparkline('html', $this.data());
    });
}
// Counters JS 
function initCounters() {
    $('.count-to').countTo();
}
//Skin changer
function skinChanger() {
    $('.right-sidebar .choose-skin li').on('click', function() {
        var $body = $('#body');
        var $this = $(this);

        var existTheme = $('.right-sidebar .choose-skin li.active').data('theme');
        $('.right-sidebar .choose-skin li').removeClass('active');
        $body.removeClass('theme-' + existTheme);
        $this.addClass('active');
        $body.addClass('theme-' + $this.data('theme'));
    });
}

// Font Setting and icon
function FontSetting() {
	$('.font_setting input:radio').click(function () {
		var others = $("[name='" + this.name + "']").map(function () {
			return this.value
		}).get().join(" ")
		console.log(others)
		$('body').removeClass(others).addClass(this.value)
	});  
}
// end

// light and dark theme setting js
var toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');
var toggleHcSwitch = document.querySelector('.theme-high-contrast input[type="checkbox"]');
var currentTheme = localStorage.getItem('theme');
if (currentTheme) {
    document.documentElement.setAttribute('data-theme', currentTheme);
  
    if (currentTheme === 'dark') {
        toggleSwitch.checked = true;
	}
	if (currentTheme === 'high-contrast') {
		toggleHcSwitch.checked = true;
		toggleSwitch.checked = false;
    }
}
function switchTheme(e) {
    if (e.target.checked) {
        document.documentElement.setAttribute('data-theme', 'dark');
		localStorage.setItem('theme', 'dark');
		$('.theme-high-contrast input[type="checkbox"]').prop("checked", false);
    }
    else {        
		document.documentElement.setAttribute('data-theme', 'light');
		localStorage.setItem('theme', 'light');
    }    
}
function switchHc(e) {
	if (e.target.checked) {
        document.documentElement.setAttribute('data-theme', 'high-contrast');
		localStorage.setItem('theme', 'high-contrast');
		$('.theme-switch input[type="checkbox"]').prop("checked", false);
    }
    else {        
		document.documentElement.setAttribute('data-theme', 'light');
		localStorage.setItem('theme', 'light');
    }  
}
toggleSwitch.addEventListener('change', switchTheme, false);
toggleHcSwitch.addEventListener('change', switchHc, false);

// end

function CustomPageJS() {
    $(".boxs-close").on('click', function(){
        var element = $(this);
        var cards = element.parents('.card');
        cards.addClass('closed').fadeOut();
    });

    $('.sub_menu_btn').on('click', function() {
        $('.sub_menu').toggleClass('show');
    });

    //Chat widget js ====
    $(document).ready(function(){
        $(".btn_overlay").on('click',function(){
            $(".overlay_menu").fadeToggle(200);
        $(this).toggleClass('btn-open').toggleClass('btn-close');
        });
    });
    $('.overlay_menu').on('click', function(){
        $(".overlay_menu").fadeToggle(200);   
        $(".overlay_menu button.btn").toggleClass('btn-open').toggleClass('btn-close');
        open = false;
    });

    //=========
    $('.form-control').on("focus", function() {
        $(this).parent('.input-group').addClass("input-group-focus");
    }).on("blur", function() {
        $(this).parent(".input-group").removeClass("input-group-focus");
    });

    // RTL version
    $(".theme-rtl input").on('change',function() {
        if(this.checked) {
            $("body").addClass('rtl_mode');
            $(".team-info, .block-header .nav-tabs, .page-calendar").addClass('rtl');
            $(".follow_us").addClass('rtl');
            $(".timeline-item").addClass('rtl');
        }else{
            $("body").removeClass('rtl_mode');
            $(".team-info, .block-header .nav-tabs, .page-calendar").removeClass('rtl');
            $(".follow_us").removeClass('rtl');
            $(".timeline-item").removeClass('rtl');
        }
    });
}


// thememakker Website live chat widget js please remove on your project
var Tawk_API=Tawk_API||{}, Tawk_LoadStart=new Date();
(function(){
var s1=document.createElement("script"),s0=document.getElementsByTagName("script")[0];
s1.async=true;
s1.src='https://embed.tawk.to/5c6d4867f324050cfe342c69/default';
s1.charset='UTF-8';
s1.setAttribute('crossorigin','*');
s0.parentNode.insertBefore(s1,s0);
})();