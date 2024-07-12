if (typeof jQuery === "undefined") {
    throw new Error("jQuery plugins need to be before this file");
}
$(function() {
    "use strict";
    $.AdminAmaze.browser.activate();
    $.AdminAmaze.leftSideBar.activate();
    $.AdminAmaze.rightSideBar.activate();
    $.AdminAmaze.navbar.activate();
    $.AdminAmaze.select.activate();

    setTimeout(function() {
        $('.page-loader-wrapper').fadeOut();
    }, 10);

    var $radios = $('input[name=menu_settings]').change(function () {
        var $body = $('body');
        var value = $radios.filter(':checked').val();
        if(value == 'menu-h') {
            $body.addClass('h_menu');

            $body.removeClass('leftmenu');
            $body.removeClass('ls-closed');
            $body.removeClass('fullwidth');

        } else if(value == 'menu-l') {
            $body.addClass('leftmenu');

            $body.removeClass('h_menu');
            $body.removeClass('ls-closed');
            $body.removeClass('fullwidth');

        } else if(value == 'menu-f') {
            $body.addClass('ls-closed');
            $body.addClass('fullwidth');

            $body.removeClass('h_menu');
            $body.removeClass('leftmenu');
        }
    });     
});
$.AdminAmaze = {};
$.AdminAmaze.options = {
    colors: {
        red: '#ec3b57',
        pink: '#E91E63',
        purple: '#ba3bd0',
        deepPurple: '#673AB7',
        indigo: '#3F51B5',
        blue: '#2196f3',
        lightBlue: '#03A9F4',
        cyan: '#00bcd4',
        green: '#4CAF50',
        lightGreen: '#8BC34A',
        yellow: '#ffe821',
        orange: '#FF9800',
        deepOrange: '#f83600',
        grey: '#9E9E9E',
        blueGrey: '#607D8B',
        black: '#000000',
        blush: '#dd5e89',
        white: '#ffffff'
    },
    leftSideBar: {
        scrollColor: 'rgba(0,0,0,0.5)',
        scrollWidth: '4px',
        scrollAlwaysVisible: false,
        scrollBorderRadius: '0',
        scrollRailBorderRadius: '0'
    },
    dropdownMenu: {
        effectIn: 'fadeIn',
        effectOut: 'fadeOut'
    }
}
/* Left Sidebar - Function */
$.AdminAmaze.leftSideBar = {
    activate: function() {
        var _this = this;
        var $body = $('body');
        var $overlay = $('.overlay');

        //Close sidebar
        $(window).on('click',function(e) {
            var $target = $(e.target);
            if (e.target.nodeName.toLowerCase() === 'i') {
                $target = $(e.target).parent();
            }

            if (!$target.hasClass('bars') && _this.isOpen() && $target.parents('#leftsidebar').length === 0) {
                if (!$target.hasClass('js-right-sidebar')) $overlay.fadeOut();
                $body.removeClass('overlay-open');
            }
        });

        $.each($('.menu-toggle.toggled'), function(i, val) {
            $(val).next().slideToggle(0);
        });

        //When page load
        $.each($('.menu .list li.active'), function(i, val) {
            var $activeAnchors = $(val).find('a:eq(0)');

            $activeAnchors.addClass('toggled');
            $activeAnchors.next().show();
        });

        //Collapse or Expand Menu
        $('.menu-toggle').on('click', function(e) {
            var $this = $(this);
            var $content = $this.next();

            if ($($this.parents('ul')[0]).hasClass('list')) {
                var $not = $(e.target).hasClass('menu-toggle') ? e.target : $(e.target).parents('.menu-toggle');

                $.each($('.menu-toggle.toggled').not($not).next(), function(i, val) {
                    if ($(val).is(':visible')) {
                        $(val).prev().toggleClass('toggled');
                        $(val).slideUp();
                    }
                });
            }

            $this.toggleClass('toggled');
            $content.slideToggle(320);
        });

        //Set menu height
        _this.checkStatuForResize(true);
        $(window).resize(function() {
            _this.checkStatuForResize(false);
        });

        //Set Waves
        Waves.attach('.menu .list a', ['waves-block']);
        Waves.init();
    },
    checkStatuForResize: function(firstTime) {
        var $body = $('body');
        var $openCloseBar = $('.navbar .navbar-header .bars');
        var width = $body.width();

        if (firstTime) {
            $body.find('.content, .sidebar').addClass('no-animate').delay(1000).queue(function() {
                $(this).removeClass('no-animate').dequeue();
            });
        }

        if (width < 1170) {
            $body.addClass('ls-closed');
            $body.removeClass('h_menu');
            $body.removeClass('leftmenu');
            $(".layout_setting_card").hide();
            $openCloseBar.fadeIn();

        } else {
            $body.removeClass('ls-closed');
            $body.addClass('h_menu');
            $(".layout_setting_card").show();
            $("input[name=menu_settings][value=menu-h]").prop('checked', true);
            $openCloseBar.fadeOut();
        }
    },
    isOpen: function() {
        return $('body').hasClass('overlay-open');
    }
};
/* Right Sidebar - Function */
$.AdminAmaze.rightSideBar = {
    activate: function() {
        var _this = this;
        var $sidebar = $('#rightsidebar');
        var $overlay = $('.overlay');

        //Close sidebar
        $(window).on('click',function(e) {
            var $target = $(e.target);
            if (e.target.nodeName.toLowerCase() === 'i') {
                $target = $(e.target).parent();
            }

            if (!$target.hasClass('js-right-sidebar') && _this.isOpen() && $target.parents('#rightsidebar').length === 0) {
                if (!$target.hasClass('bars')) $overlay.fadeOut();
                $sidebar.removeClass('open');
            }
        });

        $('.js-right-sidebar').on('click', function() {
            $sidebar.toggleClass('open');
            if (_this.isOpen()) {
                $overlay.fadeIn();
            } else {
                $overlay.fadeOut();
            }
        });
    },
    isOpen: function() {
        return $('.right-sidebar').hasClass('open');
    }
}
/* Navbar - Function ======== */
$.AdminAmaze.navbar = {
    activate: function() {
        var $body = $('body');
        var $overlay = $('.overlay');

        //Open left sidebar panel
        $('.bars').on('click', function() {
            $body.toggleClass('overlay-open');
            if ($body.hasClass('overlay-open')) {
                $overlay.fadeIn();
            } else {
                $overlay.fadeOut();
            }
        });

        //Close collapse bar on click event
        $('.nav [data-close="true"]').on('click', function() {
            var isVisible = $('.navbar-toggle').is(':visible');
            var $navbarCollapse = $('.navbar-collapse');

            if (isVisible) {
                $navbarCollapse.slideUp(function() {
                    $navbarCollapse.removeClass('in').removeAttr('style');
                });
            }
        });
    }
}
/* Form - Select - Function ======*/
$.AdminAmaze.select = {
    activate: function() {
        if ($.fn.selectpicker) {
            $('select:not(.ms)').selectpicker();
        }
    }
}
/* Browser - Function =======*/
var edge = 'Microsoft Edge';
var ie10 = 'Internet Explorer 10';
var ie11 = 'Internet Explorer 11';
var opera = 'Opera';
var firefox = 'Mozilla Firefox';
var chrome = 'Google Chrome';
var safari = 'Safari';

$.AdminAmaze.browser = {
    activate: function() {
        var _this = this;
        var className = _this.getClassName();

        if (className !== '') $('html').addClass(_this.getClassName());
    },
    getBrowser: function() {
        var userAgent = navigator.userAgent.toLowerCase();

        if (/edge/i.test(userAgent)) {
            return edge;
        } else if (/rv:11/i.test(userAgent)) {
            return ie11;
        } else if (/msie 10/i.test(userAgent)) {
            return ie10;
        } else if (/opr/i.test(userAgent)) {
            return opera;
        } else if (/chrome/i.test(userAgent)) {
            return chrome;
        } else if (/firefox/i.test(userAgent)) {
            return firefox;
        } else if (!!navigator.userAgent.match(/Version\/[\d\.]+.*Safari/)) {
            return safari;
        }

        return undefined;
    },
    getClassName: function() {
        var browser = this.getBrowser();

        if (browser === edge) {
            return 'edge';
        } else if (browser === ie11) {
            return 'ie11';
        } else if (browser === ie10) {
            return 'ie10';
        } else if (browser === opera) {
            return 'opera';
        } else if (browser === chrome) {
            return 'chrome';
        } else if (browser === firefox) {
            return 'firefox';
        } else if (browser === safari) {
            return 'safari';
        } else {
            return '';
        }
    }
}

window.Amaze= {
	colors: {

		'theme-dark1': '#343a40',
		'theme-dark2': '#636d76',
		'theme-dark3': '#939697',
		'theme-dark4': '#c7c7c7',
		'theme-dark5': '#1c1818',

		'theme-cyan1': '#59c4bc',
		'theme-cyan2': '#637aae',
		'theme-cyan3': '#2faaa1',
		'theme-cyan4': '#4cc5bc',
		'theme-cyan5': '#89bab7',

		'theme-purple1': '#7954ad',
		'theme-purple2': '#e76886',
		'theme-purple3': '#782fdf',
		'theme-purple4': '#a06ee8',
        'theme-purple5': '#a390be',
        
        'theme-blue1': '#090089',
		'theme-blue2': '#0060ca',
		'theme-blue3': '#91ceff',
		'theme-blue4': '#fcdc74',
		'theme-blue5': '#a390be',

		'theme-orange1': '#FFA901',
		'theme-orange2': '#600489',
		'theme-orange3': '#FF7F00',
		'theme-orange4': '#FBC200',
        'theme-orange5': '#38C172',
        

		'theme-a1': '#283c63',
		'theme-a2': '#f85f73',
		'theme-a3': '#928a97',
		'theme-a4': '#fbe8d3',
		'theme-a5': '#318fb5',
	},
};